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
            action_horizon=1,
            hidden_dims=mlp_cfg.hidden_dims,
            dropout=mlp_cfg.dropout,
        )
        # passing this into the residual will give a 0 residual. useful for gating
        self.null_feat = nn.Parameter(
            1e-3 * torch.randn(obs_feature_dim), requires_grad=False
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

    def solve_null_feature(self, n_steps=2000, lr=1e-2):
        """pre solve the 0 residual. should only be done once ever in training and thats it"""
        self.eval()
        with torch.no_grad():
            null_target = self.normalizer["action"].normalize(
                torch.zeros(self.action_dim, device=self.device)
            ).detach()

        self.null_feat.requires_grad = True
        opt = torch.optim.Adam([self.null_feat], lr=lr)
        loss = torch.tensor(0.0)
        for _ in range(n_steps):
            opt.zero_grad()
            null_pred = self.residual(self.null_feat.unsqueeze(0))  # (1, T, Da)
            loss = ((null_pred - null_target) ** 2).mean()
            loss.backward()
            opt.step()
        self.null_feat.requires_grad = False  # keeps it out of the main optimizer

        self.train() # go back to training
        return float(loss.item())

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
        gate = torch.sigmoid(self.gate(global_cond)).reshape(B, 1)
        feat = gate * global_cond + (1.0 - gate) * self.null_feat
        nsample = self.residual(feat)

        print(f"GATE: {gate}")
        
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

        # freeze the encoder
        with torch.set_grad_enabled(False):
            nobs_features = self.obs_encoder(nobs)["result"]
            global_cond = nobs_features.reshape(batch_size, -1)

        weights = batch["weights"] # Shape: (B, T_weight)
        if self.batched_weights:
            w = weights.mean(dim=-1) # (B,)
        else:
            w = weights[..., -1]  # (B,)

        is_correction = (w > self.correction_weight_threshold).float().unsqueeze(-1)  # (B,1)
        
        gate_logits = self.gate(global_cond).reshape(batch_size, 1)
        loss = F.binary_cross_entropy_with_logits(gate_logits, is_correction)

        with torch.no_grad():
            gate = torch.sigmoid(gate_logits)
            feat = gate * global_cond + (1.0 - gate) * self.null_feat
            pred = self.residual(feat)
            mse_loss = F.mse_loss(pred, trajectory)

            mask = is_correction.squeeze(-1) > 0.5
            loss_dict = {
                "bc_loss": mse_loss.item(),  # workspace logs this as residual_loss
                "gate_loss": loss.item(),
                "gate_mean_corr": gate[mask].mean().item() if mask.any() else float("nan"),
                "gate_mean_nocorr": gate[~mask].mean().item() if (~mask).any() else float("nan"),
            }

        return loss, loss_dict
