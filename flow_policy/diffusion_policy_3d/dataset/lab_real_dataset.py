# TODO: need to port the existing kinova dataset to this format for use with the idp3 model.
import copy
from typing import Dict

import diffusion_policy_3d.model.vision.point_process as point_process
import numpy as np
import torch
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler,
                                                downsample_mask, get_val_mask)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer)
from termcolor import cprint


class LabRealDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        has_images=True,
        has_pointclouds=False,
        weighted=False,
        num_points=4096,
    ):
        super().__init__()
        cprint(f"Loading LabRealDataset from {zarr_path}", "green")
        self.task_name = task_name

        self.has_images = has_images
        self.has_pointclouds = has_pointclouds
        self.num_points = num_points
        self.weighted = weighted

        buffer_keys = [
            "state",
            "action",
        ]

        # self.img_keys = ["front_img", "wrist_img"]
        self.img_keys = ["wrist_img"]

        if self.has_pointclouds:
            buffer_keys.append("point_cloud")
        if self.has_images:
            for key in self.img_keys:
                buffer_keys.append(key)
        if self.weighted:
            buffer_keys.append("traj_weight")

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
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

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:,].astype(np.float32)
        
        data = {
            "obs": {
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }
        
        if self.has_pointclouds:
            point_cloud = sample["point_cloud"][:,].astype(np.float32)
            point_cloud = point_process.uniform_sampling_numpy(point_cloud, self.num_points)
            data["obs"]["point_cloud"] = point_cloud
        
        if self.has_images:
            for key in self.img_keys:
                img = sample[key][:,].astype(np.float32)  # (T, H, W, C)
                img = np.transpose(img, (0, 3, 1, 2))  # Reshape to (T, C, H, W)
                data["obs"][key] = img
            
        if self.weighted:
            weights = sample['traj_weight'][:,].astype(np.float32)
            data["weights"] = weights

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: (
            torch.from_numpy(x) if x.__class__.__name__ == "ndarray" else x
        )
        torch_data = dict_apply(data, to_torch_function)
        return torch_data
