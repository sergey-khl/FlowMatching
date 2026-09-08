import h5py
import numpy as np
import os
from collections import defaultdict
import glob
import re
import torch
from scipy.spatial.transform import Rotation as R
from diffusion_policy_3d.common.pytorch_util import dict_apply

LOW_DIM_BASE_HZ = 500
IMG_BASE_HZ = 30
LOW_DIM_BASE_DT_NS = int(1e9 / LOW_DIM_BASE_HZ)
IMG_BASE_DT_NS = int(1e9 / IMG_BASE_HZ)

class Recorder:
    def __init__(self, save_dir="./dataset"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.low_dim_data = defaultdict(list)
        self.image_data = defaultdict(lambda: defaultdict(list))  # cam_name -> {"images": [...], "timestamp_ns": [...]}

    def clear(self):
        self.low_dim_data.clear()
        self.image_data.clear()

    def add_low_dim_step(self, state_dict):
        for key, value in state_dict.items():
            self.low_dim_data[key].append(value)

    def add_image_step(self, cam_name, img, timestamp_ns):
        self.image_data[cam_name]["images"].append(img)
        self.image_data[cam_name]["timestamp_ns"].append(timestamp_ns)

    def _save_recursive(self, h5_group, key, data_list):
        """Recursively saves data, creating groups for nested dictionaries."""
        # Check if the first item in the list is a dictionary
        if isinstance(data_list[0], dict):
            sub_group = h5_group.create_group(key)
            # Find all keys present in the sub-dictionaries
            sub_keys = data_list[0].keys()
            for sk in sub_keys:
                # Extract the list of values for this specific sub-key
                sub_data_list = [d[sk] for d in data_list]
                self._save_recursive(sub_group, sk, sub_data_list)
        else:
            # Base case: it's a list of arrays/numbers, so save as dataset
            data = np.array(data_list)
            print(f"SAVING episode key: {key} of size {data.shape}")
            h5_group.create_dataset(key, data=data, compression="gzip")

    def get_next_episode_index(self):
        pattern = os.path.join(self.save_dir, "episode_*.hdf5")
        existing = glob.glob(pattern)

        max_idx = -1
        regex = re.compile(r"_(\d+)\.hdf5$")

        for filepath in existing:
            filename = os.path.basename(filepath)
            match = regex.search(filename)
            if match:
                idx = int(match.group(1))
                max_idx = max(max_idx, idx)

        return max_idx + 1

    def save_episode(self, episode_name):
        if not self.low_dim_data or not self.image_data:
            print("one or both of low_dim / image is missing, not saving.")
            return

        filepath = os.path.join(self.save_dir, f"{episode_name}.hdf5")
        with h5py.File(filepath, "w") as f:
            if self.low_dim_data:
                low_dim_group = f.create_group("low_dim")
                for key, data_list in self.low_dim_data.items():
                    self._save_recursive(low_dim_group, key, data_list)

            if self.image_data:
                images_group = f.create_group("images")
                for cam_name, cam_dict in self.image_data.items():
                    cam_group = images_group.create_group(cam_name)
                    imgs = np.array(cam_dict["images"])
                    print(f"SAVING episode key: images/{cam_name}/images of size {imgs.shape}")
                    cam_group.create_dataset("images", data=imgs, compression="gzip")

                    ts_name = cam_name.replace("_image", "_img") + "_timestamp_ns"
                    ts_data = np.array(cam_dict["timestamp_ns"], dtype=np.int64)
                    print(f"SAVING episode key: images/{cam_name}/{ts_name} of size {ts_data.shape}")
                    cam_group.create_dataset(ts_name, data=ts_data, compression="gzip")

        print(f"[RECORDER] Saved {episode_name} to {filepath}")
        self.clear()

    def _quat_to_rotmat(self, quat_wxyz):
        """Quaternion [w, x, y, z] -> first two rows of the rotation matrix"""
        quat_xyzw = np.roll(np.asarray(quat_wxyz), -1, axis=-1)
        rotmat = R.from_quat(quat_xyzw).as_matrix()
        return rotmat[:2, :].reshape(6)

    def _downsample_grid_indices(self, timestamps_ns, target_dt_ns):
        """global downsample. fullfilling role of sparse_query_frequency_down_sample_steps"""
        timestamps_ns = np.asarray(timestamps_ns)
        t0, t1 = timestamps_ns[0], timestamps_ns[-1]
        if t1 <= t0:
            return np.array([0], dtype=np.int64)

        target_times = np.arange(t0, t1 + 1, target_dt_ns)
        idxs = np.searchsorted(timestamps_ns, target_times)
        idxs = np.clip(idxs, 0, len(timestamps_ns) - 1)
        left = np.clip(idxs - 1, 0, len(timestamps_ns) - 1)
        use_left = np.abs(timestamps_ns[left] - target_times) < np.abs(timestamps_ns[idxs] - target_times)
        idxs = np.where(use_left, left, idxs)
        return np.unique(idxs).astype(np.int64)

    def _match_nearest_indices(self, query_timestamps, ref_timestamps):
        """sync 2 diff data stereams"""
        query_timestamps = np.asarray(query_timestamps)
        ref_timestamps = np.asarray(ref_timestamps)
        idxs = np.searchsorted(ref_timestamps, query_timestamps)
        idxs = np.clip(idxs, 1, len(ref_timestamps) - 1)
        left = ref_timestamps[idxs - 1]
        right = ref_timestamps[idxs]
        idxs = idxs - (np.abs(query_timestamps - left) < np.abs(query_timestamps - right))
        return idxs

    def _gather_raw_low_dim(self, key, raw_ids):
        if key == "ext_torque":
            # key = "filtered_human_torque"
            key = "filtered_environment_torque"
        elif key == "policy_cmd":
            key = "policy_jp"
        data = np.asarray(self.low_dim_data[key], dtype=np.float32)
        return data[raw_ids]

    def _gather_state(self, raw_ids):
        leader_jp = np.asarray(self.low_dim_data["leader_jp"])            # (N,7)
        leader_cart_pos = np.asarray(self.low_dim_data["leader_cart_pos"])  # (N,3)
        leader_quat = np.asarray(self.low_dim_data["leader_quat_wxyz"])     # (N,4)
        gripper_pos = np.asarray(self.low_dim_data["gripper_pos"])
        rows = [
            np.concatenate([
                leader_jp[i],
                leader_cart_pos[i],
                self._quat_to_rotmat(leader_quat[i]),
                # [gripper_pos[i]],
            ]).astype(np.float32)
            for i in raw_ids
        ]
        return np.stack(rows, axis=0)  # (horizon, 16)

    def _gather_images(self, cam_data, raw_ids):
        frames = [
            cam_data["images"][i].transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
            for i in raw_ids
        ]
        return np.stack(frames, axis=0)  # (horizon, 3, 224, 224)

    def get_obs_dict(self, obs_key_horizons: dict, device="cuda", dtype=torch.float32):
        """
        Build a batched obs_dict from buffered episode data, anchored on the most recent front_img
        """
        front = self.image_data.get("front_img")
        wrist = self.image_data.get("wrist_img")
        if not front or not front["images"] or not wrist or not wrist["images"]:
            print("no images")
            return None
        if not self.low_dim_data.get("timestamp_ns") or not self.low_dim_data.get("leader_jp"):
            print("no low dim")
            return None

        front_ts_raw = np.asarray(front["timestamp_ns"])
        wrist_ts_raw = np.asarray(wrist["timestamp_ns"])
        low_dim_ts_raw = np.asarray(self.low_dim_data["timestamp_ns"])

        # global downsample based on front image ts as anchor
        front_ds_idx = self._downsample_grid_indices(front_ts_raw, IMG_BASE_DT_NS)
        front_ts_ds = front_ts_raw[front_ds_idx]

        # sync wrist images to fron image.
        wrist_ds_idx = self._match_nearest_indices(front_ts_ds, wrist_ts_raw)

        # same thing for low dim
        low_dim_ds_idx = self._downsample_grid_indices(low_dim_ts_raw, LOW_DIM_BASE_DT_NS)
        low_dim_ts_ds = low_dim_ts_raw[low_dim_ds_idx]

        img_center = len(front_ts_ds) - 1
        anchor_time = front_ts_ds[img_center]

        low_dim_center = int(self._match_nearest_indices(
            np.array([anchor_time]), low_dim_ts_ds
        )[0])

        # get downsapmled horizon worth of each obs and action item
        obs_result = {}
        for key, cfg in obs_key_horizons.items():
            horizon = cfg["horizon"]
            down_sample_steps = cfg.get("down_sample_steps", 1)
            source = cfg.get("source", "low_dim")
            span = (horizon - 1) * down_sample_steps

            if source == "low_dim":
                if low_dim_center < span:
                    return None
                ds_ids = np.arange(low_dim_center - span, low_dim_center + 1, down_sample_steps)
                raw_ids = low_dim_ds_idx[ds_ids]
                if key == "state":
                    obs_result[key] = self._gather_state(raw_ids)
                else:
                    obs_result[key] = self._gather_raw_low_dim(key, raw_ids)
            elif source == "img":  # img
                if img_center < span:
                    return None
                ds_ids = np.arange(img_center - span, img_center + 1, down_sample_steps)
                if key == "front_img":
                    raw_ids = front_ds_idx[ds_ids]
                    obs_result[key] = self._gather_images(front, raw_ids)
                elif key == "wrist_img":
                    raw_ids = wrist_ds_idx[ds_ids]
                    obs_result[key] = self._gather_images(wrist, raw_ids)
                else:
                    raise ValueError(f"get_obs_dict: unrecognized image obs key '{key}'")

        return dict_apply(
            obs_result,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=dtype),
        )

    
    def get_example_obs_dict(self, obs_key_horizons: dict, device="cuda", dtype=torch.float32):
        """
        Build a dummy batched obs_dict for a dry run of the policy.
        """
        obs_result = {}
        for key, cfg in obs_key_horizons.items():
            horizon = cfg["horizon"]
            source = cfg.get("source", "low_dim")
            
            if source == "low_dim":
                if key == "state":
                    # 7 (jp) + 3 (cart_pos) + 6 (rotmat) + 1 (gripper) = 17
                    shape = (1, horizon, 16)
                elif key in ["leader_jp", "policy_cmd", "filtered_human_torque", "filtered_environment_torque", "ext_torque"]:
                    shape = (1, horizon, 7)
            elif source == "weight":
                continue
            elif source == "img":
                shape = (1, horizon, 3, 224, 224)
            else:
                raise ValueError(f"get_example_obs_dict: unrecognized source '{source}' for key '{key}'")
                
            obs_result[key] = torch.zeros(shape, device=device, dtype=dtype)
            
        return obs_result

