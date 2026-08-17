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

CHUNK_SIZE = 100

POS_THRESHOLD = 0.0005
ROT_THRESHOLD = 0.0002

GRIPPER_ACTION_THRESHOLD = -0.15

IMPORTANCE_POWER = 1  #  changes the spread of the traj_weight
 
# TODO: set 50
TARGET_FPS = 10
TARGET_DT_NS = int(1e9 / TARGET_FPS)  # 100_000_000 ns

IMG_TARGET_FPS = 10
IMG_TARGET_DT_NS = int(1e9 / IMG_TARGET_FPS)

def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))

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
    img_size = 224

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image


if __name__ == "__main__":
    # expert_data_paths = ["/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_no_dmp", "/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_no_dmp2"]
    # expert_data_paths = ["/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_dmp", "/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_dmp2"]
    expert_data_paths = ["/mnt/10tb/serg/datasets/pandp_dmp2"]
    # expert_data_paths = ["/home/serg/projects/wam-ros-docker/wam_ws/src/wam_haptic_dmps/pandp_no_dmp2"]
    save_data_path = f"/mnt/10tb/serg/FlowMatching/data_wam/wam_pandp-solo-cup_dmp-100-{IMPORTANCE_POWER}_jp-ee.zarr"

    dirs = []
    for path in expert_data_paths:
        files = os.listdir(path)
        files = sorted(
            [f for f in files if re.match(r"episode_.*\.h(?:df)?5$", f)],
            key=lambda x: int(re.search(r"episode_(\d+).*\.h(?:df)?5$", x).group(1))
        )
        dirs.extend([(path, f) for f in files])

    # storage
    total_count = 0
    front_img_arrays = []
    wrist_img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    traj_weight_arrays = []
    ext_torque_arrays = []

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
    
    # check the first episode to see if we have images and point clouds
    data = h5py.File(os.path.join(dirs[0][0], dirs[0][1]), "r")

    for i, (base_path, demo_dir) in enumerate(dirs):
        dir_name = os.path.dirname(demo_dir)

        cprint("Processing {}".format(demo_dir), "green")
        
        data = h5py.File(os.path.join(base_path, demo_dir), "r")
        low_dim = data["low_dim"]

        timestamps_ns = low_dim["timestamp_ns"][:]
        jp = low_dim["leader_jp"][:]       # (N, 7)
        cart_pos = low_dim["leader_cart_pos"][:]       # (N, 3)
        cart_rot = low_dim["leader_quat_wxyz"][:]        # (N, 4) w,x,y,z
        gripper_pos = low_dim["gripper_pos"][:]           # (N,)
        policy_torque_scale = low_dim["policy_torque_scale"][:] # (N, 7)
        front_image = data["front_image"]                 # (N, H, W, 3)
        wrist_image = data["wrist_image"]
        ext_torque = low_dim["leader_ext_torque"][:]        # (N, 4) w,x,y,z

        t0, t1 = timestamps_ns[0], timestamps_ns[-1]
        target_times = np.arange(t0, t1 + 1, TARGET_DT_NS)
        idxs = np.array([find_nearest_index(timestamps_ns, t) for t in target_times], dtype=np.int64)
        idxs = np.unique(idxs)  # sorted, deduped

        jp_ds = jp[idxs]
        cart_pos_ds = cart_pos[idxs]
        cart_rot_ds = cart_rot[idxs]
        gripper_pos_ds = gripper_pos[idxs]
        front_img_ds = front_image[idxs]
        wrist_img_ds = wrist_image[idxs]
        policy_torque_scale_ds = policy_torque_scale[idxs]
        ext_torque_ds = ext_torque[idxs]

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
                [gripper_pos_ds[j]],
                # [-gripper_pos_ds[j]],
            ]).astype(np.float32)  # (8,)
 
            action = np.concatenate([
                # cart_pos_ds[j + 1],
                # quats_ds[j + 1],
                # delta_pos,
                jp_ds[j + 1],
                ext_torque_ds[j + 1],
                # [gripper_action],
                [gripper_action],
            ]).astype(np.float32)  # (7,)
 
            if has_weights:
                weight_scale = len(idxs) / weight_sum
                episode_weight_arrays.append(np.sum(inverted_torques[j], axis=0) * weight_scale)

            state_arrays.append(state)
            action_arrays.append(action)
            front_img_arrays.append(preprocess_image(front_img_ds[j]))
            wrist_img_arrays.append(preprocess_image(wrist_img_ds[j]))
            ext_torque_arrays.append(ext_torque_ds[j])
 
            prev_gripper_action = gripper_action
            episode_len += 1
 
        if episode_len > 0:
            noop_percentage = (noop_count / (episode_len + noop_count)) * 100
            print(f"[{demo_dir}]  Noops: {noop_count}/{(episode_len + noop_count)} steps ({noop_percentage:.2f}%) removed.")
        else:
            print(f"[{demo_dir}]  Episode too short to form a single step.")
 
        if has_weights:
            episode_weights = np.array(episode_weight_arrays, dtype=np.float64)
            total = episode_weights.sum()
            spread = episode_weights ** IMPORTANCE_POWER
            spread_sum = spread.sum()
            episode_weights = spread * (total / spread_sum)
            traj_weight_arrays.extend(episode_weights)

        episode_ends_arrays.append(total_count + episode_len)
        total_count += episode_len
        

    # create zarr file
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    if has_images:
        front_img_arrays = np.stack(front_img_arrays, axis=0)
        wrist_img_arrays = np.stack(wrist_img_arrays, axis=0)
        if front_img_arrays.shape[1] == 3:  # make channel last
            front_img_arrays = np.transpose(front_img_arrays, (0, 2, 3, 1))
        if wrist_img_arrays.shape[1] == 3:  # make channel last
            wrist_img_arrays = np.transpose(wrist_img_arrays, (0, 2, 3, 1))
            
    if has_pointclouds:
        point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    
    if has_weights:
        traj_weight_arrays = np.stack(traj_weight_arrays, axis=0)

    action_arrays = np.stack(action_arrays, axis=0)
    state_arrays = np.stack(state_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    ext_torque_arrays = np.stack(ext_torque_arrays, axis=0)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    if has_images:
        front_img_chunk_size = (
            CHUNK_SIZE,
            front_img_arrays.shape[1],
            front_img_arrays.shape[2],
            front_img_arrays.shape[3],
        )
        wrist_img_chunk_size = (
            CHUNK_SIZE,
            wrist_img_arrays.shape[1],
            wrist_img_arrays.shape[2],
            wrist_img_arrays.shape[3],
        )
        
    if has_pointclouds:
        point_cloud_chunk_size = (
            CHUNK_SIZE,
            point_cloud_arrays.shape[1],
            point_cloud_arrays.shape[2],
        )

    if has_weights:
        traj_weights_chunk_size = (
            CHUNK_SIZE,
        )
    
    # depth_chunk_size = (CHUNK_SIZE, depth_arrays.shape[1], depth_arrays.shape[2])
    if len(action_arrays.shape) == 2:
        action_chunk_size = (CHUNK_SIZE, action_arrays.shape[1])
    elif len(action_arrays.shape) == 3:
        action_chunk_size = (CHUNK_SIZE, action_arrays.shape[1], action_arrays.shape[2])
    else:
        raise NotImplementedError

    # NOTE: assuming 7dof robot
    ext_torque_chunk_size = (CHUNK_SIZE, 7)
    
    if has_images:
        zarr_data.create_dataset(
            "front_img",
            data=front_img_arrays,
            chunks=front_img_chunk_size,
            dtype="uint8",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "wrist_img",
            data=wrist_img_arrays,
            chunks=wrist_img_chunk_size,
            dtype="uint8",
            overwrite=True,
            compressor=compressor,
        )
    
    if has_pointclouds:
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            dtype="float64",
            overwrite=True,
            compressor=compressor,
        )
        
    if has_weights:
        zarr_data.create_dataset('traj_weight',
             data=traj_weight_arrays,
             dtype='float32',
             chunks=traj_weights_chunk_size,
             overwrite=True,
             compressor=compressor
        )

    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(CHUNK_SIZE, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        chunks=(CHUNK_SIZE,),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "ext_torque",
        data=ext_torque_arrays,
        chunks=ext_torque_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )


    # print shape
    if has_images:
        cprint(
            f"front img shape: {front_img_arrays.shape}, range: [{np.min(front_img_arrays)}, {np.max(front_img_arrays)}]",
            "green",
        )
        cprint(
            f"wrist img shape: {wrist_img_arrays.shape}, range: [{np.min(wrist_img_arrays)}, {np.max(wrist_img_arrays)}]",
            "green",
        )
        
    if has_pointclouds:
        cprint(
            f"point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]",
            "green",
        )

    if has_weights:
        cprint(
            f"weight shape: {traj_weight_arrays.shape}, range: [{np.min(traj_weight_arrays)}, {np.max(traj_weight_arrays)}]",
            "green",
        )
        
    cprint(
        f"action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]",
        "green",
    )
    cprint(
        f"state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]",
        "green",
    )
    cprint(
        f"episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]",
        "green",
    )
    cprint(
        f"ext_torque shape: {ext_torque_arrays.shape}, range: [{np.min(ext_torque_arrays)}, {np.max(ext_torque_arrays)}]",
        "green",
    )
    cprint(f"total_count: {total_count}", "green")
    cprint(f"Saved zarr file to {save_data_path}", "green")
