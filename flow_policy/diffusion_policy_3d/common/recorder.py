import h5py
import numpy as np
import os
from collections import defaultdict
import glob
import re
import torch
from scipy.spatial.transform import Rotation as R
from diffusion_policy_3d.common.pytorch_util import dict_apply

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
        """Quaternion [w, x, y, z] -> first two rows of the rotation matrix, flattened (6,)."""
        quat_xyzw = np.roll(np.asarray(quat_wxyz), -1, axis=-1)
        rotmat = R.from_quat(quat_xyzw).as_matrix()
        return rotmat[:2, :].reshape(6)

    def _nearest_low_dim_idx(self, query_time):
        low_dim_time = np.asarray(self.low_dim_data["timestamp_ns"])
        idx = int(np.searchsorted(low_dim_time, query_time))
        return min(idx, len(low_dim_time) - 1)

    def _nearest_img_idx(self, cam_name, query_time):
        ts = np.asarray(self.image_data[cam_name]["timestamp_ns"])
        idx = int(np.searchsorted(ts, query_time))
        return min(idx, len(ts) - 1)

    def _slice_state_backward(self, anchor_time, horizon, down_sample_steps):
        center_idx = self._nearest_low_dim_idx(anchor_time)
        span = (horizon - 1) * down_sample_steps
        if center_idx < span:
            return None  # not enough history... dont infer

        ids = np.arange(center_idx - span, center_idx + 1, down_sample_steps)

        leader_jp = np.asarray(self.low_dim_data["leader_jp"])            # (N,7)
        leader_cart_pos = np.asarray(self.low_dim_data["leader_cart_pos"])  # (N,3)
        leader_quat = np.asarray(self.low_dim_data["leader_quat_wxyz"])     # (N,4)

        rows = [
            np.concatenate([
                leader_jp[i],
                leader_cart_pos[i],
                self._quat_to_rotmat(leader_quat[i]),
            ]).astype(np.float32)
            for i in ids
        ]
        return np.stack(rows, axis=0)  # (horizon, 16)

    def _slice_img_backward(self, cam_name, anchor_time, horizon, down_sample_steps):
        cam_data = self.image_data[cam_name]
        center_idx = self._nearest_img_idx(cam_name, anchor_time)
        span = (horizon - 1) * down_sample_steps
        if center_idx < span:
            return None  # not enough history... dont infer

        ids = np.arange(center_idx - span, center_idx + 1, down_sample_steps)

        frames = [
            cam_data["images"][i].transpose(2, 0, 1).astype(np.float32)
            for i in ids
        ]
        return np.stack(frames, axis=0)  # (horizon, 3, 224, 224)

    def get_obs_dict(self, obs_key_horizons: dict, device="cuda", dtype=torch.float32):
        """
        Build a batched obs_dict from buffered episode data, anchored on the most recent front_img
        """
        front = self.image_data.get("front_img")
        wrist = self.image_data.get("wrist_img")
        if not front or not front["images"] or not wrist or not wrist["images"]:
            return None
        if not self.low_dim_data.get("timestamp_ns") or not self.low_dim_data.get("leader_jp"):
            return None

        anchor_time = front["timestamp_ns"][-1]

        obs_result = {}
        for key, cfg in obs_key_horizons.items():
            horizon = cfg["horizon"]
            down_sample_steps = cfg.get("down_sample_steps", 1)
            source = cfg.get("source", "low_dim")

            if source == "low_dim":
                sliced = self._slice_state_backward(anchor_time, horizon, down_sample_steps)
            else:
                sliced = self._slice_img_backward(key, anchor_time, horizon, down_sample_steps)

            if sliced is None:
                return None

            obs_result[key] = sliced

        return dict_apply(
            obs_result,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=dtype),
        )

