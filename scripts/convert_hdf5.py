import os
import re

import h5py
import numpy as np
import torch
import torchvision
import tqdm
import zarr
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

has_images = True
has_pointclouds = False
has_weights = True
has_policy = True

CHUNK_SIZE = 100

POS_THRESHOLD = 0.0005
ROT_THRESHOLD = 0.0002

GRIPPER_ACTION_THRESHOLD = -0.15

IMPORTANCE_POWER = 1  #  changes the spread of the traj_weight
 
TARGET_FPS = 500
TARGET_DT_NS = int(1e9 / TARGET_FPS)  # 100_000_000 ns

IMG_TARGET_FPS = 30
IMG_TARGET_DT_NS = int(1e9 / IMG_TARGET_FPS)  # 100_000_000 ns

def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))

def match_nearest_indices(query_timestamps, ref_timestamps):
    """
    For each timestamp in query_timestamps, find the index into ref_timestamps whose value is closest.
    """
    idxs = np.searchsorted(ref_timestamps, query_timestamps)
    idxs = np.clip(idxs, 1, len(ref_timestamps) - 1)

    left = ref_timestamps[idxs - 1]
    right = ref_timestamps[idxs]

    # step back to idxs-1 if that's actually closer
    idxs -= (np.abs(query_timestamps - left) < np.abs(query_timestamps - right))
    return idxs

def fetch_by_indices(dataset, idxs):
    """
    indexing with duplicates
    """
    unique_idxs, inverse = np.unique(idxs, return_inverse=True)
    fetched = dataset[unique_idxs]        # h5py read: unique & increasing, OK
    return fetched[inverse]               # numpy read: dups/any order, OK

def quats_to_rotvec(quats):
    """compute state angle as a rotvec based on the home quat"""
    quats_xyzw = np.roll(quats, -1, axis=-1)
    ref_xyzw = np.roll(np.array([0.8311519102829725, 0.0005888003896006177, -0.556030721230029, -0.003999049322114239]), -1 , axis=-1) # home quat
    r = R.from_quat(ref_xyzw).inv() * R.from_quat(quats_xyzw)
    return r.as_rotvec()

def quats_to_euler(quats):
    """Quaternion [w, x, y, z] → Euler (roll, pitch, yaw) in radians."""
    quats_xyzw = np.roll(quats, -1, axis=-1)

    euler = R.from_quat(quats_xyzw).as_euler('xyz', degrees=False)  # (N,3)
    return np.unwrap(euler, axis=0)

def quats_to_rotmat(quats):
    """Quaternion [w, x, y, z] -> first two rows of the rotation matrix."""
    quats_xyzw = np.roll(quats, -1, axis=-1)
    rotmat = R.from_quat(quats_xyzw).as_matrix()  # (..., 3, 3)
    return rotmat[..., :2, :].reshape(*rotmat.shape[:-2], 6)

def quats_to_quat(quats):
    """Quaternion [w, x, y, z] → quat"""
    ref_quat = np.array([0.8311519102829725, 0.0005888003896006177, -0.556030721230029, -0.003999049322114239]) # home quat
    for i in range(len(quats)):
        if np.dot(quats[i], ref_quat) < 0:
            quats[i] *= -1
    return quats

def delta_rotvec(rot_current, rot_next):
    """Minimal rotation delta between two quaternions, as an axis-angle (rotation) vector."""
    curr_xyzw = np.roll(rot_current, -1, axis=-1)
    next_xyzw = np.roll(rot_next, -1, axis=-1)
    r_delta = R.from_quat(curr_xyzw).inv() * R.from_quat(next_xyzw)
    return r_delta.as_rotvec()  # shape (3,), axis * angle_rad

def delta_euler(rot_current, rot_next):
    """Minimal rotation delta between two euler angles. already unwraped"""
    r_delta = rot_next - rot_current
    return r_delta

def delta_quats(q1, q2):
    """Rotation from q1 to q2 (q2 * inverse(q1), with q1 assumed unit norm)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 + x1*x2 + y1*y2 + z1*z2,
        w1*x2 - x1*w2 - y1*z2 + z1*y2,
        w1*y2 + x1*z2 - y1*w2 - z1*x2,
        w1*z2 - x1*y2 + y1*x2 - z1*w2,
    ])

def delta_jp(jp_current, jp_next):
    """just sub the jp's"""
    jp_delta = jp_next - jp_current
    return jp_delta

def is_noop(delta_pos, delta_rot, gripper_action, prev_gripper_action):
    """
    port from      https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py   """
    motion_is_noop = (np.linalg.norm(delta_pos) < POS_THRESHOLD and
                       np.linalg.norm(delta_rot) < ROT_THRESHOLD)
 
    if prev_gripper_action is None:
        return motion_is_noop
 
    return motion_is_noop and (gripper_action == prev_gripper_action)


def zero_noop_axes(delta_pos, delta_rot):
    """Zero out individual axes of a delta that don't clear their threshold"""
    delta_pos = np.where(np.abs(delta_pos) > POS_THRESHOLD, delta_pos, 0.0)
    delta_rot = np.where(np.abs(delta_rot) > ROT_THRESHOLD, delta_rot, 0.0)
    return delta_pos, delta_rot


def preprocess_image(image):
    img_size = (224, 224)

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, img_size)
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def downsample_by_timestamp(timestamps_ns, target_dt_ns):
    t0, t1 = timestamps_ns[0], timestamps_ns[-1]
    target_times = np.arange(t0, t1 + 1, target_dt_ns)
    idxs = np.array(
        [find_nearest_index(timestamps_ns, t) for t in target_times], dtype=np.int64
    )
    return np.unique(idxs)

def append_ds(group, name, data, chunk_shape, dtype):
    """Create a resizable dataset on first call, append on every call after."""
    if name not in group:
        group.create_dataset(
            name, shape=(0,) + data.shape[1:], chunks=chunk_shape,
            dtype=dtype, compressor=compressor,
        )
    group[name].append(data, axis=0)

if __name__ == "__main__":
    # expert_data_paths = ["/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_no_dmp", "/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_no_dmp2"]
    # expert_data_paths = ["/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_dmp", "/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_dmp2"]
    # expert_data_paths = ["/mnt/10tb/serg/datasets/pandp_dmp2"]
    expert_data_paths = ["/mnt/10tb/serg/datasets/plug_asus_153"]
    save_data_path = f"/mnt/10tb/serg/FlowMatching/data_wam/wam_plug-asus-153_nocorr-{IMPORTANCE_POWER}.zarr"

    dirs = []
    for path in expert_data_paths:
        files = os.listdir(path)
        files = sorted(
            [f for f in files if re.match(r"episode_.*\.h(?:df)?5$", f)],
            key=lambda x: int(re.search(r"episode_(\d+).*\.h(?:df)?5$", x).group(1))
        )
        dirs.extend([(path, f) for f in files])

    # raw robot data
    total_count = 0
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    robot_timestamp_arrays = []

    # imgs
    img_total_count = 0
    front_img_arrays = []
    wrist_img_arrays = []
    img_episode_ends_arrays = []
    image_timestamp_arrays = []

    # extras. follow raw robot hz
    policy_cmd_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    human_torque_arrays = []
    traj_weight_arrays = []

    if os.path.exists(save_data_path):
        cprint("Data already exists at {}".format(save_data_path), "red")
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = "y"
        if user_input == "y":
            cprint("Overwriting {}".format(save_data_path), "red")
            os.system("rm -rf {}".format(save_data_path))
        else:
            cprint("Exiting", "red")
            exit()
    os.makedirs(save_data_path, exist_ok=True)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_root = zarr.group(save_data_path)
    low_dim_root = zarr_root.create_group("low_dim")
    low_dim_data = low_dim_root.create_group("data")
    low_dim_meta = low_dim_root.create_group("meta")
    img_root = zarr_root.create_group("img")
    img_data = img_root.create_group("data")
    img_meta = img_root.create_group("meta")
    
    # check the first episode to see if we have images and point clouds
    data = h5py.File(os.path.join(dirs[0][0], dirs[0][1]), "r")

    for i, (base_path, demo_dir) in enumerate(dirs):
        dir_name = os.path.dirname(demo_dir)

        
        data = h5py.File(os.path.join(base_path, demo_dir), "r")
        low_dim = data["low_dim"]
        images = data["images"]

        if has_images and ("front_image" not in images or "wrist_image" not in images):
            cprint(f"[{demo_dir}] Missing front_image/wrist_image, skipping episode", "yellow")
            data.close()
            continue

        cprint("Processing {}".format(demo_dir), "green")

        timestamps_ns = low_dim["timestamp_ns"][:]
        jp = low_dim["leader_jp"][:]       # (N, 7)
        cart_pos = low_dim["leader_cart_pos"][:]       # (N, 3)
        cart_rot = low_dim["leader_quat_wxyz"][:]        # (N, 4) w,x,y,z
        gripper_pos = low_dim["gripper_pos"][:]           # (N,)
        policy_torque_scale = low_dim["policy_torque_scale"][:] # (N, 7)
        policy_cmd = low_dim["policy_jp"][:] # (N, 7)
        human_torque = low_dim["human_torque"][:]        # (N, 4) w,x,y,z

        # robot and images captured with different unix time thing
        front_timestamps_ns_raw = images["front_image"]["front_img_timestamp_ns"][:]
        clock_offset_ns = int(front_timestamps_ns_raw[0]) - int(timestamps_ns[0])
        timestamps_ns = timestamps_ns + clock_offset_ns

        idxs = downsample_by_timestamp(timestamps_ns, TARGET_DT_NS)

        jp_ds = jp[idxs]
        cart_pos_ds = cart_pos[idxs]
        cart_rot_ds = cart_rot[idxs]
        gripper_pos_ds = gripper_pos[idxs]
        policy_torque_scale_ds = policy_torque_scale[idxs]
        policy_cmd_ds = policy_cmd[idxs]
        human_torque_ds = human_torque[idxs]
        robot_timestamp_ds = timestamps_ns[idxs]

        # rotvec_ds = quats_to_rotvec(cart_rot_ds)
        # euler_ds = quats_to_euler(cart_rot_ds)
        # quats_ds = quats_to_quat(cart_rot_ds)
        rotmats_ds = quats_to_rotmat(cart_rot_ds)
        gripper_action_ds = np.where(gripper_pos_ds > GRIPPER_ACTION_THRESHOLD, 1, -1)

        weight_sum = 0
        if has_weights:
            # smooth torque scale across the episode (time axis) per-joint
            window_length = min(11, len(idxs))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)

            if window_length > polyorder:
                policy_torque_scale_ds = savgol_filter(
                    policy_torque_scale_ds, window_length=window_length,
                    polyorder=polyorder, axis=0
                )

            inverted_torques = 1 - policy_torque_scale_ds[:, 0:4]
            for j in range(len(idxs)):
                weight_sum += np.sum(inverted_torques[j], axis=0)


        episode_len = 0
        prev_gripper_action = None
        noop_count = 0
        episode_weight_arrays = []

        # raw robot data comes in at 500hz
        for j in range(len(idxs) - 1):
            # delta_pos = cart_pos_ds[j + 1] - cart_pos_ds[j]
            # delta_rot = delta_rotvec(cart_rot_ds[j], cart_rot_ds[j + 1])
            # delta_rot = delta_euler(euler_ds[j], euler_ds[j + 1])
            # delta_rot = delta_quats(quats_ds[j], quats_ds[j + 1])
            # delta_rot = delta_jp(jp_ds[j], jp_ds[j + 1])
            gripper_action = gripper_action_ds[j + 1]
 
            # if is_noop(delta_pos, delta_rot, gripper_action, prev_gripper_action):
            #     noop_count += 1
            #     continue
 
            state = np.concatenate([
                jp_ds[j],
                cart_pos_ds[j],
                rotmats_ds[j],
                # policy_cmd_ds[j],
                # ext_torque_ds[j],
                # [gripper_pos_ds[j]],
            ]).astype(np.float32)  # (8,)
 
            action = np.concatenate([
                # cart_pos_ds[j + 1],
                # quats_ds[j + 1],
                # delta_pos,
                jp_ds[j + 1],
                human_torque_ds[j + 1],
                # [gripper_action],
            ]).astype(np.float32)  # (7,)
 
            if has_weights:
                weight_scale = len(idxs) / weight_sum
                episode_weight_arrays.append(np.sum(inverted_torques[j], axis=0) * weight_scale)

            if has_policy:
                policy_cmd_arrays.append(policy_cmd_ds[j].astype(np.float32))

            state_arrays.append(state)
            action_arrays.append(action)
            human_torque_arrays.append(human_torque_ds[j])
            robot_timestamp_arrays.append(robot_timestamp_ds[j])
 
            prev_gripper_action = gripper_action
            episode_len += 1

        if episode_len > 0:
            ep_state = np.stack(state_arrays, axis=0)
            ep_action = np.stack(action_arrays, axis=0)
            ep_human_torque = np.stack(human_torque_arrays, axis=0)
            ep_robot_ts = np.array(robot_timestamp_arrays, dtype=np.int64)

            append_ds(low_dim_data, "state", ep_state, (CHUNK_SIZE, ep_state.shape[1]), "float32")
            append_ds(low_dim_data, "action", ep_action, (CHUNK_SIZE, ep_action.shape[1]), "float32")
            append_ds(low_dim_data, "ext_torque", ep_human_torque, (CHUNK_SIZE, 7), "float32")
            append_ds(low_dim_data, "robot_timestamp_ns", ep_robot_ts, (CHUNK_SIZE,), "int64")

            if has_policy:
                ep_policy = np.stack(policy_cmd_arrays, axis=0)
                append_ds(low_dim_data, "policy_cmd", ep_policy, (CHUNK_SIZE, 7), "float32")

            if has_weights:
                episode_weights = np.array(episode_weight_arrays, dtype=np.float64)
                total = episode_weights.sum()
                spread = episode_weights ** IMPORTANCE_POWER
                episode_weights = spread * (total / spread.sum())
                ep_weights = episode_weights.astype(np.float32)
                append_ds(low_dim_data, "traj_weight", ep_weights, (CHUNK_SIZE,), "float32")

            # reset per-episode lists so they don't accumulate across episodes
            state_arrays, action_arrays = [], []
            human_torque_arrays, robot_timestamp_arrays = [], []
            policy_cmd_arrays = []
 
        # NOTE: noop does not work with the sepeartion of the images
        if episode_len > 0:
            noop_percentage = (noop_count / (episode_len + noop_count)) * 100
            print(f"[{demo_dir}]  Noops: {noop_count}/{(episode_len + noop_count)} steps ({noop_percentage:.2f}%) removed.")
        else:
            print(f"[{demo_dir}]  Episode too short to form a single step.")
 

        episode_ends_arrays.append(total_count + episode_len)
        total_count += episode_len

        if has_images:
            front_image = images["front_image"]["images"]  # (M, H, W, 3)
            wrist_image = images["wrist_image"]["images"]   # (K, H, W, 3)

            front_timestamps_ns = images["front_image"]["front_img_timestamp_ns"][:]
            wrist_timestamps_ns = images["wrist_image"]["wrist_img_timestamp_ns"][:]

            # downsample using front camera as the anchor
            img_idxs = downsample_by_timestamp(front_timestamps_ns, IMG_TARGET_DT_NS)
            front_img_ds = front_image[img_idxs]
            image_timestamp_ds = front_timestamps_ns[img_idxs]

            # find the wrist frame closest in time to each chosen front frame
            wrist_idxs = match_nearest_indices(image_timestamp_ds, wrist_timestamps_ns)
            wrist_img_ds = fetch_by_indices(wrist_image, wrist_idxs)
            
            ep_front = np.stack([preprocess_image(front_img_ds[j]) for j in range(len(img_idxs))], axis=0)
            ep_wrist = np.stack([preprocess_image(wrist_img_ds[j]) for j in range(len(img_idxs))], axis=0)
            ep_img_ts = np.array(image_timestamp_ds, dtype=np.int64)

            if ep_front.shape[1] == 3:
                ep_front = np.transpose(ep_front, (0, 2, 3, 1))
            if ep_wrist.shape[1] == 3:
                ep_wrist = np.transpose(ep_wrist, (0, 2, 3, 1))

            front_chunk = (CHUNK_SIZE, ep_front.shape[1], ep_front.shape[2], ep_front.shape[3])
            wrist_chunk = (CHUNK_SIZE, ep_wrist.shape[1], ep_wrist.shape[2], ep_wrist.shape[3])

            append_ds(img_data, "front_img", ep_front, front_chunk, "uint8")
            append_ds(img_data, "wrist_img", ep_wrist, wrist_chunk, "uint8")
            append_ds(img_data, "img_timestamp_ns", ep_img_ts, (CHUNK_SIZE,), "int64")

            img_episode_len = len(img_idxs)
            img_episode_ends_arrays.append(img_total_count + img_episode_len)
            img_total_count += img_episode_len

    if has_images:
        img_meta.create_dataset(
            "episode_ends",
            data=img_episode_ends_arrays,
            chunks=(CHUNK_SIZE,),
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
    
    low_dim_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        chunks=(CHUNK_SIZE,),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    cprint(f"total_count: {total_count}", "green")
    cprint(f"Saved zarr file to {save_data_path}", "green")
