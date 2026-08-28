import copy
import math
import time
from typing import Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.vision.timm_obs_encoder_with_force import TimmObsEncoderWithForce
from diffusion_policy_3d.model.compliant_residual.mlp import MLP
from diffusion_policy_3d.policy.base_policy import BasePolicy
from einops import rearrange, reduce
from termcolor import cprint


class CrDaggerRGB(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        mlp_cfg,
        obs_as_global_cond=True,
        condition_type="film",
        use_depth=False,
        use_depth_only=False,
        obs_encoder=None,
        weighted_loss=False,
        batched_weights=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type

        assert (
            obs_as_global_cond is True
        ), "only implemented global conditioning for flow matching as of now"

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }

        # create the Image encoder from config.
        self.use_depth = use_depth
        self.use_depth_only = use_depth_only
        obs_encoder: TimmObsEncoderWithForce = obs_encoder

        self.rgb_keys = obs_encoder.rgb_keys
        self.low_dim_keys = obs_encoder.low_dim_keys
        
        # create diffusion model
        if use_depth and not use_depth_only:
            for key in self.rgb_keys:
                obs_shape_meta[key]['shape'][0] = 4 # 3,H,W -> 4,H,W
        
        if use_depth and use_depth_only:
            for key in self.rgb_keys:
                obs_shape_meta[key]['shape'][0] = 1

        obs_feature_dim = np.prod(obs_encoder.output_shape())

        self.residual = MLP(
            input_dim=obs_feature_dim,
            action_dim= action_dim,
            action_horizon=mlp_cfg.action_horizon,
            hidden_dims=mlp_cfg.hidden_dims,
            dropout=mlp_cfg.dropout,
        )
        self.obs_encoder = obs_encoder

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.kwargs = kwargs
        self.weighted_loss = weighted_loss
        self.batched_weights = batched_weights

        print_params(self)

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        for key in self.rgb_keys:
            nobs[key] /= 255.0
            if nobs[key].shape[-1] == 3:
                if len(nobs[key].shape) == 5:
                    nobs[key] = nobs[key].permute(0, 1, 4, 2, 3)
                if len(nobs[key].shape) == 4:
                    nobs[key] = nobs[key].permute(0, 3, 1, 2)
            if self.use_depth and not self.use_depth_only:
                nobs[key] = torch.cat([nobs[key], nobs['depth'].unsqueeze(-3)], dim=-3)
            if self.use_depth and self.use_depth_only:
                nobs[key] = nobs['depth'].unsqueeze(-3)


        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        nobs_features = self.obs_encoder(nobs)["result"]
        global_cond = nobs_features.reshape(B, -1)

        nsample = self.residual(global_cond)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # get prediction
        result = {
            "action": action,
            "action_pred": action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        for key in self.rgb_keys:
            nobs[key] /= 255.0
            if nobs[key].shape[-1] == 3:
                if len(nobs[key].shape) == 5:
                    nobs[key] = nobs[key].permute(0, 1, 4, 2, 3)
                if len(nobs[key].shape) == 4:
                    nobs[key] = nobs[key].permute(0, 3, 1, 2)
            if self.use_depth and not self.use_depth_only:
                nobs[key] = torch.cat([nobs[key], nobs['depth'].unsqueeze(-3)], dim=-3)
            if self.use_depth and self.use_depth_only:
                nobs[key] = nobs['depth'].unsqueeze(-3)

        batch_size = nactions.shape[0]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions

        nobs_features = self.obs_encoder(nobs)["result"]
        global_cond = nobs_features.reshape(batch_size, -1)

        target = trajectory

        weights = batch["weights"]  # (B, T)

        pred = self.residual(global_cond)  # (B,1) or (B,T)
        if pred.shape != target.shape:
            pred = pred.expand_as(target)
        loss = F.mse_loss(pred, target)

        loss = loss
        
        if self.weighted_loss:
            weights = batch["weights"].unsqueeze(-1)
            loss = weights * loss
        
        # loss = reduce(loss, "b ... -> b (...)", "mean")
        # loss = loss.mean()
        loss = loss.sum()

        loss_dict = {
            "bc_loss": loss.item(),
        }

        return loss, loss_dict
