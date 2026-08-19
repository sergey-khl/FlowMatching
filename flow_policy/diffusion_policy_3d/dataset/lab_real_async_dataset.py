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
        obs_key_horizons: dict,
        extra_streams: dict = None,
        action_horizon: int = 8,
        action_down_sample_steps: int = 1,
        action_padding: bool = False,
        low_dim_time_key: str = "robot_timestamp_ns",
        query_frequency_down_sample_steps: int = 1,
        correction_upsampling: dict = None,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        has_pointclouds=False,
        weighted=False,
        num_points=4096,
    ):
        super().__init__()
        cprint(f"Loading LabRealDataset from {zarr_path}", "green")
        self.task_name = task_name

        self.has_pointclouds = has_pointclouds
        self.num_points = num_points
        self.weighted = weighted
        self.obs_key_horizons = obs_key_horizons
        self.action_horizon = action_horizon
        self.action_down_sample_steps = action_down_sample_steps
        self.action_padding = action_padding
        self.low_dim_time_key = low_dim_time_key
        self.extra_streams = extra_streams or {}
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
        if self.weighted:
            low_dim_keys.add("traj_weight")
        # TODO: pc should be sep
        if self.has_pointclouds:
            low_dim_keys.add("point_cloud")
        for key, cfg in self.obs_key_horizons.items():
            if cfg.get("source", "low_dim") == "low_dim":
                low_dim_keys.add(key)
        self.low_dim_replay_buffer = ReplayBuffer.copy_from_path(
            os.path.join(zarr_path, "low_dim"), keys=list(low_dim_keys)
        )

        used_sources = {
            cfg.get("source", "low_dim") for cfg in obs_key_horizons.values()
        }

        self.source_replay_buffers = {}
        for source_name in used_sources:
            if source_name == "low_dim":
                continue
            stream_cfg = self.extra_streams[source_name]
            time_key = stream_cfg["time_key"]
            source_keys = {
                key
                for key, cfg in obs_key_horizons.items()
                if cfg.get("source") == source_name
            }
            source_keys.add(time_key)
 
            source_path = os.path.join(zarr_path, source_name)
            assert os.path.exists(source_path), (
                f"obs_key_horizons references source='{source_name}' but it gone :("
            )
            self.source_replay_buffers[source_name] = ReplayBuffer.copy_from_path(
                source_path, keys=list(source_keys)
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
        extra_sources = {
            name: {
                "replay_buffer": rb,
                "time_key": self.extra_streams[name]["time_key"],
            }
            for name, rb in self.source_replay_buffers.items()
        }
        return SequenceSamplerAsync(
            low_dim_replay_buffer=self.low_dim_replay_buffer,
            obs_key_horizons=self.obs_key_horizons,
            action_horizon=self.action_horizon,
            action_down_sample_steps=self.action_down_sample_steps,
            action_padding=self.action_padding,
            low_dim_time_key=self.low_dim_time_key,
            extra_sources=extra_sources,
            episode_mask=episode_mask,
            query_frequency_down_sample_steps=self.query_frequency_down_sample_steps,
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
        data = {"action": self.replay_buffer["action"], "agent_pos": self.replay_buffer["state"][..., :]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if self.has_pointclouds:
            normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
            
        if self.has_images:
            for key in self.img_keys:
                normalizer[key] = SingleFieldLinearNormalizer.create_identity()
            
        if self.weighted:
            normalizer["weights"] = SingleFieldLinearNormalizer.create_identity()

        normalizer["ext_torque"] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, obs_sample, action):
        data = {"obs": {}, "action": action.astype(np.float32)}
 
        for key, cfg in self.obs_key_horizons.items():
            arr = obs_sample[key].astype(np.float32)
            out_key = "agent_pos" if key == "state" else key
            if cfg.get("source") == "img":
                arr = np.transpose(arr, (0, 3, 1, 2))  # (T,H,W,C) -> (T,C,H,W)
            if cfg.get("source") == "point_cloud":
                arr = point_process.uniform_sampling_numpy(arr, self.num_points)
            if cfg.get("source") == "traj_weight":
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
