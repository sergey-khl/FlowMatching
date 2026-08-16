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
from diffusion_policy_3d.model.diffusion.mask_generator import \
    LowdimMaskGenerator
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import \
    ConditionalUnet1D
from diffusion_policy_3d.model.vision.timm_obs_encoder_with_force import TimmObsEncoderWithForce
from diffusion_policy_3d.model.guidance.mlp import MLP
from diffusion_policy_3d.policy.base_policy import BasePolicy
from einops import rearrange, reduce
from termcolor import cprint


class DgDaggerRGB(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        mlp_cfg,
        num_inference_steps=10,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        use_depth=False,
        use_depth_only=False,
        obs_encoder=None,
        weighted_loss=False,
        batched_weights=False,
        correction_thresh=1.0,
        max_delay=4,
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

        self.scale_predictor = MLP(
            input_dim=obs_feature_dim,
            action_horizon=mlp_cfg.action_horizon,
            hidden_dims=mlp_cfg.hidden_dims,
            dropout=mlp_cfg.dropout,
        )

        self.base_unet = ConditionalUnet1D(
            input_dim=action_dim + 1,  # add +1 for time step as extra input.
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.corrected_unet = ConditionalUnet1D(
            input_dim=action_dim + 1,  # add +1 for time step as extra input.
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.weighted_loss = weighted_loss
        self.batched_weights = batched_weights
        self.max_delay = max_delay

        self.num_inference_steps = num_inference_steps

        print_params(self)

    # ========= inference  ============
    def conditional_sample(
        self,
        action_prefix,
        condition_data,
        condition_mask,
        condition_data_pc=None,
        condition_mask_pc=None,
        local_cond=None,
        global_cond=None,
        generator=None,
        delay=0,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        # Start from random noise.
        trajectory = torch.normal(
            0,
            1,
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )
        prefix_mask = torch.arange(self.horizon)[None, :] < delay  # (1, horizon)
        prefix_mask = prefix_mask.to(trajectory.device)
        
        B = condition_data.shape[0]

        delta = 1.0 / self.num_inference_steps
        for t in torch.arange(0, 1, delta):
            trajectory = torch.where(
                prefix_mask.unsqueeze(-1), action_prefix, trajectory
            )

            # Get time encoding for this step
            tau_tensor = torch.full(
                (B,), t, device=condition_data.device, dtype=torch.float
            )

            # Concatenate the time mask to the input, to tell the model
            # which part is the prefix.
            tau_map = tau_tensor.view(B, 1, 1).expand(B, self.horizon, 1)
            tau_masked = torch.where(
                prefix_mask.unsqueeze(-1), torch.ones_like(tau_map), tau_map
            )
            model_input = torch.cat([trajectory, tau_masked], dim=-1)

            base_step = self.base_unet(
                sample=model_input, timestep=tau_tensor,
                local_cond=local_cond, global_cond=global_cond,
            )[..., : self.action_dim]

            corrected_step = self.corrected_unet(
                sample=model_input, timestep=tau_tensor,
                local_cond=local_cond, global_cond=global_cond,
            )[..., : self.action_dim]

            correction_scale = F.softplus(self.scale_predictor(global_cond))  # (B,1) or (B,T)
            if correction_scale.dim() == 2 and correction_scale.shape[1] == 1:
                correction_scale = correction_scale.view(B, 1, 1)
            else:
                correction_scale = correction_scale.view(B, self.horizon, 1)

            step = base_step + correction_scale * (corrected_step - base_step)

            step = step[..., : self.action_dim]  # only keep action dim

            # Euler integration step
            trajectory = trajectory - delta * step
        
        # finally, enforce prefix
        trajectory = torch.where(
            prefix_mask.unsqueeze(-1), action_prefix, trajectory
        )

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], past_actions: torch.Tensor = None
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
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...]
            )
            nobs_features = self.obs_encoder(this_nobs)["result"]
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...]
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # TODO: pad the action prefix to the correct size.
        action_prefix = torch.zeros(
            (B, self.horizon, self.action_dim),
            device=device,
            dtype=dtype
        )
        
        if past_actions is not None and past_actions.shape[1] > 0:
            # Insert valid history into the start of the buffer
            past_actions_normalized = self.normalizer["action"].normalize(past_actions)
            current_delay = past_actions.shape[1]
            action_prefix[:, :past_actions.shape[1], :] = past_actions_normalized[:, :past_actions.shape[1], :]
        else:
            current_delay = 0

        # run sampling via euler integration
        nsample = self.conditional_sample(
            action_prefix=action_prefix,
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            delay=current_delay,
            **self.kwargs,
        )

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
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...]
            )
            nobs_features = self.obs_encoder(this_nobs)["result"]

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x)
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # this is just 0's
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image (in range 0, 1), uniform sampled.
        global_timesteps = torch.rand(bsz, device=trajectory.device).float()

        # Interpolate trajectory and noise using the timesteps
        # sample random delay, then generate a mask to not compute loss for that prefix.
        if self.max_delay > 0:
            delay = torch.randint(
                0, self.max_delay, (bsz,), device=trajectory.device
            ).unsqueeze(
                -1
            )  # (bsz, 1)
        else:
            delay = torch.zeros((bsz, 1), device=trajectory.device, dtype=torch.long)
        prefix_mask = (
            torch.arange(horizon)[None, :].to(trajectory.device) < delay
        )  # (bsz, horizon)
        local_timesteps = torch.where(prefix_mask, 1.0, global_timesteps.unsqueeze(-1))
        local_timesteps = local_timesteps.unsqueeze(-1)  # (bsz, horizon, 1)

        # Add noise, but NOT to the prefix
        noisy_trajectory = local_timesteps * trajectory + (1 - local_timesteps) * noise

        # compute loss mask
        loss_mask = ~condition_mask

        # Don't compute loss on the prefix region
        loss_mask = loss_mask & (~prefix_mask.unsqueeze(-1))

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # add the extra dimension for time step
        model_input = torch.cat(
            [noisy_trajectory, local_timesteps], dim=-1
        )  # (bsz, horizon, action_dim + 1)

        # Predict the vector field
        base_pred = self.base_unet(
            sample=model_input, timestep=global_timesteps,
            local_cond=local_cond, global_cond=global_cond,
        )[..., : self.action_dim]

        corrected_pred = self.corrected_unet(
            sample=model_input, timestep=global_timesteps,
            local_cond=local_cond, global_cond=global_cond,
        )[..., : self.action_dim]

        target = noise - trajectory

        weights = batch["weights"]  # (B, T)

        if self.batched_weights:
            is_correction = (weights > self.correction_thresh).unsqueeze(-1)  # (B, T, 1)
            scale_target = weights  # (B, T)
        else:  # per_chunk — per-sample trajectory mean, not batch-wide
            chunk_weight = weights.mean(dim=1, keepdim=True)                  # (B, 1)
            is_correction = (chunk_weight > self.correction_thresh)
            is_correction = is_correction.unsqueeze(-1).expand(-1, horizon, -1)  # (B, T, 1)
            scale_target = chunk_weight.expand(-1, horizon)                    # (B, T)

        base_loss_mask = loss_mask & (~is_correction)
        corr_loss_mask = loss_mask & is_correction

        base_loss = (F.mse_loss(base_pred, target, reduction="none") * base_loss_mask
                     ).sum() / base_loss_mask.sum().clamp_min(1.0)
        corr_loss = (F.mse_loss(corrected_pred, target, reduction="none") * corr_loss_mask
                     ).sum() / corr_loss_mask.sum().clamp_min(1.0)

        pred_scale = F.softplus(self.scale_predictor(global_cond))  # (B,1) or (B,T)
        if pred_scale.shape != scale_target.shape:
            pred_scale = pred_scale.expand_as(scale_target)
        scale_loss = F.mse_loss(pred_scale, scale_target)

        loss = base_loss + corr_loss + self.scale_loss_weight * scale_loss

        loss = loss * loss_mask.type(loss.dtype)
        
        if self.weighted_loss:
            weights = batch["weights"].unsqueeze(-1)
            loss = weights * loss
        
        # loss = reduce(loss, "b ... -> b (...)", "mean")
        # loss = loss.mean()
        valid_elements_count = loss_mask.sum().clamp_min(1.0)
        loss = loss.sum() / valid_elements_count

        loss_dict = {
            "bc_loss": (base_loss + corr_loss).item(),
            "base_loss": base_loss.item(),
            "corrected_loss": corr_loss.item(),
            "scale_loss": scale_loss.item(),
        }

        return loss, loss_dict
