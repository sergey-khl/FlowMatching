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


USING_JOINTS = False

POS_THRESHOLD = 0.0005
ROT_THRESHOLD = 0.0002

GRIPPER_ACTION_THRESHOLD = -0.01
 
TARGET_FPS = 10
TARGET_DT_NS = int(1e9 / TARGET_FPS)  # 100_000_000 ns

def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))

def quats_to_rotvec(quats):
    """compute state angle as a rotvec based on the home quat"""
    quats_xyzw = np.roll(quats, -1, axis=-1)
    ref_xyzw = np.roll(np.array([0.8311519102829725, 0.0005888003896006177, -0.556030721230029, -0.003999049322114239]), -1 , axis=-1) # home quat
    r = R.from_quat(ref_xyzw).inv() * R.from_quat(quats_xyzw)
    return r.as_rotvec()


def delta_rotvec(rot_current, rot_next):
    """Minimal rotation delta between two quaternions, as an axis-angle (rotation) vector."""
    curr_xyzw = np.roll(rot_current, -1, axis=-1)
    next_xyzw = np.roll(rot_next, -1, axis=-1)
    r_delta = R.from_quat(curr_xyzw).inv() * R.from_quat(next_xyzw)
    return r_delta.as_rotvec()  # shape (3,), axis * angle_rad


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
    img_size = 84

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image


if __name__ == "__main__":
    expert_data_path = "/home/serg/projects/openpi-wam/dataset"
    save_data_path = "/home/serg/projects/FlowMatching/flow_policy/data_wam/reach_solo_cup.zarr"

    dirs = os.listdir(expert_data_path)
    dirs = sorted(
        [f for f in dirs if re.match(r"episode_.*\.h(?:df)?5$", f)],
        key=lambda x: int(re.search(r"episode_(\d+).*\.h(?:df)?5$", x).group(1))
    )

    # storage
    total_count = 0
    front_img_arrays = []
    wrist_img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

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
    data = h5py.File(os.path.join(expert_data_path, dirs[0]), "r")
    has_images = "front_image" in data.keys()
    has_pointclouds = "pointcloud" in data.keys()

    takeover_combined = []

    for i, demo_dir in enumerate(dirs):
        dir_name = os.path.dirname(demo_dir)

        cprint("Processing {}".format(demo_dir), "green")
        
        data = h5py.File(os.path.join(expert_data_path, demo_dir), "r")
        low_dim = data["low_dim"]

        timestamps_ns = low_dim["timestamp_ns"][:]
        cart_pos = low_dim["follower_cart_pos"][:]       # (N, 3)
        cart_rot = low_dim["follower_cart_rot"][:]        # (N, 4) w,x,y,z
        gripper_pos = low_dim["gripper_pos"][:]           # (N,)
        front_image = data["front_image"]                 # (N, H, W, 3)
        wrist_image = data["wrist_image"]

        t0, t1 = timestamps_ns[0], timestamps_ns[-1]
        target_times = np.arange(t0, t1 + 1, TARGET_DT_NS)
        idxs = np.array([find_nearest_index(timestamps_ns, t) for t in target_times], dtype=np.int64)
        idxs = np.unique(idxs)  # sorted, deduped

        cart_pos_ds = cart_pos[idxs]
        cart_rot_ds = cart_rot[idxs]
        gripper_pos_ds = gripper_pos[idxs]
        front_img_ds = front_image[idxs]
        wrist_img_ds = wrist_image[idxs]

        rotvec_ds = quats_to_rotvec(cart_rot_ds)
        gripper_action_ds = np.where(gripper_pos_ds > GRIPPER_ACTION_THRESHOLD, 1, -1)

        episode_len = 0
        prev_gripper_action = None
        noop_count = 0

        for j in range(len(idxs) - 1):
            delta_pos = cart_pos_ds[j + 1] - cart_pos_ds[j]
            delta_rot = delta_rotvec(cart_rot_ds[j], cart_rot_ds[j + 1])
            gripper_action = gripper_action_ds[j + 1]
 
            if is_noop(delta_pos, delta_rot, gripper_action, prev_gripper_action):
                noop_count += 1
                continue
 
            state = np.concatenate([
                cart_pos_ds[j],
                rotvec_ds[j],
                [gripper_pos_ds[j]],
                [-gripper_pos_ds[j]],
            ]).astype(np.float32)  # (8,)
 
            action = np.concatenate([
                delta_pos,
                delta_rot,
                [gripper_action],
            ]).astype(np.float32)  # (7,)
 
            state_arrays.append(state)
            action_arrays.append(action)
            front_img_arrays.append(preprocess_image(front_img_ds[j]))
            wrist_img_arrays.append(preprocess_image(wrist_img_ds[j]))
 
            prev_gripper_action = gripper_action
            episode_len += 1
 
        if episode_len > 0:
            noop_percentage = (noop_count / (episode_len + noop_count)) * 100
            print(f"[{demo_dir}]  Noops: {noop_count}/{(episode_len + noop_count)} steps ({noop_percentage:.2f}%) removed.")
        else:
            print(f"[{demo_dir}]  Episode too short to form a single step.")
 
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
    
    action_arrays = np.stack(action_arrays, axis=0)
    state_arrays = np.stack(state_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    if has_images:
        front_img_chunk_size = (
            100,
            front_img_arrays.shape[1],
            front_img_arrays.shape[2],
            front_img_arrays.shape[3],
        )
        wrist_img_chunk_size = (
            100,
            wrist_img_arrays.shape[1],
            wrist_img_arrays.shape[2],
            wrist_img_arrays.shape[3],
        )
        
    if has_pointclouds:
        point_cloud_chunk_size = (
            100,
            point_cloud_arrays.shape[1],
            point_cloud_arrays.shape[2],
        )
    
    # depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    if len(action_arrays.shape) == 2:
        action_chunk_size = (100, action_arrays.shape[1])
    elif len(action_arrays.shape) == 3:
        action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
    else:
        raise NotImplementedError
    
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
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        chunks=(100,),
        dtype="int64",
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
    cprint(f"total_count: {total_count}", "green")
    cprint(f"Saved zarr file to {save_data_path}", "green")
