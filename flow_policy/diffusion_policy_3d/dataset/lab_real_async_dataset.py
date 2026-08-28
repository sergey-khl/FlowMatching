# TODO: need to port the existing kinova dataset to this format for use with the idp3 model.
import copy
from typing import Dict
import os

import diffusion_policy_3d.model.vision.point_process as point_process
import numpy as np
import torch
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (downsample_mask, get_val_mask)
from diffusion_policy_3d.common.sampler_async import SequenceSamplerAsync
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer)
from termcolor import cprint


class LabRealAsyncDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        shape_meta: dict,
        action_padding: bool = False,
        low_dim_time_key: str = "robot_timestamp_ns",
        img_time_key: str = "img_timestamp_ns",
        query_frequency_down_sample_steps: int = 1,
        correction_upsampling: dict = None,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        has_pointclouds=False,
        has_policy=False,
        num_points=4096,
    ):
        super().__init__()
        cprint(f"Loading LabRealDataset from {zarr_path}", "green")
        self.task_name = task_name

        self.has_pointclouds = has_pointclouds
        self.has_policy = has_policy
        self.num_points = num_points

        sample_cfg = shape_meta["sample"]
        self.obs_key_horizons = sample_cfg["obs"]["sparse"]
        action_cfg = sample_cfg["action"]["sparse"]
        self.action_horizon = action_cfg["horizon"]
        self.action_down_sample_steps = action_cfg.get(
            "down_sample_steps", 1
        )
        self.buffer_time_ms = sample_cfg.get(
            "training_duration_per_sparse_query", 200
        )

        self.action_padding = action_padding
        self.low_dim_time_key = low_dim_time_key
        self.img_time_key = img_time_key
        self.query_frequency_down_sample_steps = query_frequency_down_sample_steps
        self.correction_upsampling = {
            "enabled": False,
            "threshold": 1.0,
            "lookback_horizon": 1,
            "lookback_down_sample_steps": 1,
            "weighted_sampling": 1,
            "correction_horizon": 0,
            "only_correction": False,
            **(correction_upsampling or {}),
        }

        low_dim_keys = {"state", "action", low_dim_time_key}
        if self.correction_upsampling["enabled"]:
            low_dim_keys.add("traj_weight")
        # TODO: pc should be sep
        if self.has_pointclouds:
            low_dim_keys.add("point_cloud")

        img_keys = {img_time_key}
        for key, cfg in self.obs_key_horizons.items():
            source = cfg.get("source", "low_dim")
            if source == "low_dim" or source == "weight":
                low_dim_keys.add(key)
            elif source == "img":
                img_keys.add(key)
            else:
                raise ValueError(
                    f"obs_key_horizons['{key}'].source == '{source}' but only "
                    f"'low_dim' and 'img' are supported"
                )

        self.low_dim_replay_buffer = ReplayBuffer.copy_from_path(
            os.path.join(zarr_path, "low_dim"), keys=list(low_dim_keys)
        )
        self.img_replay_buffer = ReplayBuffer.copy_from_path(
            os.path.join(zarr_path, "img"), keys=list(img_keys)
        )

        val_mask = get_val_mask(
            n_episodes=self.low_dim_replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        self.train_mask = train_mask
 
        self.sampler = self._make_sampler(train_mask)

    def _make_sampler(self, episode_mask):
        return SequenceSamplerAsync(
            low_dim_replay_buffer=self.low_dim_replay_buffer,
            img_replay_buffer=self.img_replay_buffer,
            obs_key_horizons=self.obs_key_horizons,
            action_horizon=self.action_horizon,
            action_down_sample_steps=self.action_down_sample_steps,
            action_padding=self.action_padding,
            low_dim_time_key=self.low_dim_time_key,
            img_time_key=self.img_time_key,
            episode_mask=episode_mask,
            query_frequency_down_sample_steps=self.query_frequency_down_sample_steps,
            buffer_time_ms=self.buffer_time_ms,
            correction_upsampling=self.correction_upsampling["enabled"],
            correction_weight_threshold=self.correction_upsampling["threshold"],
            correction_lookback_horizon=self.correction_upsampling["lookback_horizon"],
            correction_lookback_down_sample_steps=self.correction_upsampling["lookback_down_sample_steps"],
            weighted_sampling=self.correction_upsampling["weighted_sampling"],
            correction_horizon=self.correction_upsampling["correction_horizon"],
            only_correction=self.correction_upsampling["only_correction"],
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = self._make_sampler(~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.low_dim_replay_buffer["action"],
            "state": self.low_dim_replay_buffer["state"][..., :],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
 
        img_out_keys = [
            key for key, cfg in self.obs_key_horizons.items()
            if cfg.get("source") == "img"
        ]
        for key in img_out_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
 
        if self.has_pointclouds:
            normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()

        if self.has_policy:
            normalizer["policy_cmd"] = SingleFieldLinearNormalizer.create_identity()
 
        normalizer["weights"] = SingleFieldLinearNormalizer.create_identity()
 
        normalizer["ext_torque"] = SingleFieldLinearNormalizer.create_identity()

 
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, obs_sample, action):
        data = {"obs": {}, "action": action.astype(np.float32)}
 
        for key, cfg in self.obs_key_horizons.items():
            arr = obs_sample[key].astype(np.float32)
            out_key = key
            if cfg.get("source") == "img":
                arr = np.transpose(arr, (0, 3, 1, 2))  # (T,H,W,C) -> (T,C,H,W)
            if cfg.get("source") == "point_cloud":
                arr = point_process.uniform_sampling_numpy(arr, self.num_points)
            if cfg.get("source") == "weight":
                data["weights"] = arr
                continue
            data["obs"][out_key] = arr
        
            
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs_sample, action = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(obs_sample, action)
        to_torch_function = lambda x: (
            torch.from_numpy(x) if x.__class__.__name__ == "ndarray" else x
        )
        torch_data = dict_apply(data, to_torch_function)
        return torch_data
