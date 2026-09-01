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


class GcrDaggerRGB(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        mlp_cfg,
        correction_weight_threshold,
        obs_as_global_cond=True,
        condition_type="film",
        use_depth=False,
        use_depth_only=False,
        obs_encoder=None,
        weighted_loss=False,
        batched_weights=False,
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type
        self.correction_weight_threshold = correction_weight_threshold

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
        
        self.use_depth = use_depth
        self.use_depth_only = use_depth_only
        obs_encoder: TimmObsEncoderWithForce = obs_encoder

        self.rgb_keys = obs_encoder.rgb_keys
        self.low_dim_keys = obs_encoder.low_dim_keys
        
        if use_depth and not use_depth_only:
            for key in self.rgb_keys:
                obs_shape_meta[key]['shape'][0] = 4 
        
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
        
        # Predicts a scalar gate value per step to indicate correction
        self.gate = MLP(
            input_dim=obs_feature_dim,
            action_dim=1, 
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

        cprint("Freezing base residual and obs_encoder networks for gCrDagger post-training.", "yellow")
        for param in self.residual.parameters():
            param.requires_grad = False
        for param in self.obs_encoder.parameters():
            param.requires_grad = False

        print_params(self)

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

        device = self.device
        dtype = self.dtype

        nobs_features = self.obs_encoder(nobs)["result"]
        global_cond = nobs_features.reshape(B, -1)

        # --- GATED PREDICTION ---
        residual_out = self.residual(global_cond)
        gate_logits = self.gate(global_cond)
        gate_val = torch.sigmoid(gate_logits) # Shape: (B, T, 1)
        
        # Broadcast gate over the residual dimensions
        nsample = gate_val * residual_out

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        result = {
            "action": action_pred,
            "action_pred": action_pred,
        }
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
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
        trajectory = nactions

        # Only compute gradients for the gating network if freeze_base is True
        with torch.set_grad_enabled(False):
            nobs_features = self.obs_encoder(nobs)["result"]
            global_cond = nobs_features.reshape(batch_size, -1)
            pred_residual = self.residual(global_cond)

        gate_logits = self.gate(global_cond)
        gate_val = torch.sigmoid(gate_logits)
        
        pred = gate_val * pred_residual
        target = trajectory

        mse_loss = F.mse_loss(pred, target)

        weights = batch["weights"] # Shape: (B, T_weight)
        is_correction_current = (weights[:, -1] > self.correction_weight_threshold).float() # Shape: (B,)
        
        if gate_logits.dim() == 2:
            is_correction_target = is_correction_current.unsqueeze(-1) # (B, 1)
        else:
            is_correction_target = is_correction_current.view(-1, 1, 1).expand_as(gate_logits)

        gate_loss = F.binary_cross_entropy(gate_val, is_correction_target)
        
        # only backpropagate through the gate loss
        loss = gate_loss
        
        loss_dict = {
            "bc_loss": mse_loss.item(), # Log original metric for tracking
            "gate_loss": gate_loss.item(),
        }

        return loss, loss_dict
