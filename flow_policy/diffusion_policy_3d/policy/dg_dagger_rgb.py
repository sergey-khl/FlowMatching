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
        vision_feature_dim = int(np.prod(obs_encoder.vision_output_shape()))
        torque_feature_dim = int(np.prod(obs_encoder.torque_output_shape()))
        low_dim_feature_dim = int(np.prod(obs_encoder.low_dim_output_shape()))

        vision_global_dim = vision_feature_dim + low_dim_feature_dim
        torque_global_dim = torque_feature_dim + low_dim_feature_dim

        self.scale_predictor = MLP(
            input_dim=obs_feature_dim,
            action_horizon=mlp_cfg.action_horizon,
            hidden_dims=mlp_cfg.hidden_dims,
            dropout=mlp_cfg.dropout,
        )

        self.gate = MLP(
            input_dim=obs_feature_dim,
            action_horizon=1,
            hidden_dims=mlp_cfg.hidden_dims, # Use your config's hidden dims
            dropout=mlp_cfg.dropout,
        )

        self.vision_unet = ConditionalUnet1D(
            input_dim=action_dim + 1,  # add +1 for time step as extra input.
            local_cond_dim=None,
            global_cond_dim=vision_global_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.torque_unet = ConditionalUnet1D(
            input_dim=action_dim + 1,  # add +1 for time step as extra input.
            local_cond_dim=None,
            global_cond_dim=torque_global_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.null_feat = nn.Parameter(
            1e-3 * torch.randn(obs_feature_dim), requires_grad=False
        )

        self.obs_encoder = obs_encoder

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.kwargs = kwargs
        self.weighted_loss = weighted_loss
        self.batched_weights = batched_weights
        self.correction_thresh = correction_thresh
        self.max_delay = max_delay

        self.num_inference_steps = num_inference_steps

        print_params(self)

    def solve_null_feature(self, n_steps=2000, lr=1e-2):
        """pre calc null feature so scale_predictor outputs 0."""
        self.eval()
        self.null_feat.requires_grad = True
        opt = torch.optim.Adam([self.null_feat], lr=lr)
        loss = torch.tensor(0.0)
        for _ in range(n_steps):
            opt.zero_grad()
            # scale_predictor expects (B, obs_feature_dim)
            null_pred = F.softplus(self.scale_predictor(self.null_feat.unsqueeze(0)))
            loss = (null_pred ** 2).mean()
            loss.backward()
            opt.step()
        self.null_feat.requires_grad = False
        self.train()
        return float(loss.item())

    # ========= inference  ============
    def conditional_sample(
        self,
        action_prefix,
        condition_data,
        enc_dict,
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

        vision_feat = enc_dict["vision_feature"]
        torque_feat = enc_dict["torque_feature"]
        low_dim_feat = enc_dict["low_dim_feature"]
        full_feat = enc_dict["result"]

        gate_logits = self.gate(full_feat).reshape(B, 1)
        gate = torch.sigmoid(gate_logits)

        vision_unet_cond = torch.cat([vision_feat, low_dim_feat], dim=-1)
        gated_torque_feat = gate * torque_feat
        torque_unet_cond = torch.cat([gated_torque_feat, low_dim_feat], dim=-1)
        print(f"GATE: {gate}")
        scale_cond = gate * full_feat + (1 - gate) * self.null_feat

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

            vision_step = self.vision_unet(sample=model_input, timestep=tau_tensor, local_cond=None, global_cond=vision_unet_cond)[..., :self.action_dim]
            torque_step = self.torque_unet(sample=model_input, timestep=tau_tensor, local_cond=None, global_cond=torque_unet_cond)[..., :self.action_dim]


            correction_scale = F.softplus(self.scale_predictor(scale_cond))  # (B,1) or (B,T)
            if correction_scale.dim() == 2 and correction_scale.shape[1] == 1:
                correction_scale = correction_scale.view(B, 1, 1)
            else:
                correction_scale = correction_scale.view(B, self.horizon, 1)

            print(f"CORRECTION SCALE: {correction_scale[0][0]}")

            step = vision_step + correction_scale * (torque_step - vision_step)

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

        # build input
        device = self.device
        dtype = self.dtype

        enc_dict = self.obs_encoder(nobs)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype) # basically just used for finding traj shape

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
            enc_dict=enc_dict,
            delay=current_delay,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        # start = To - 1
        # end = start + self.n_action_steps
        # action = action_pred[:, start:end]
        #
        # get prediction
        result = {
            # "action": action,
            "action": action_pred,
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

        B = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        enc_dict = self.obs_encoder(nobs)
        vision_feat = enc_dict["vision_feature"]
        torque_feat = enc_dict["torque_feature"]
        low_dim_feat = enc_dict["low_dim_feature"]
        full_feat = enc_dict["result"]

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

        # indicator function taking in vision + torque + low_dim
        gate_logits = self.gate(full_feat).reshape(B, 1)
        gate = torch.sigmoid(gate_logits) # (B, 1)

        # gated features just like in paper.
        gated_torque_feat = gate * torque_feat 
        torque_unet_cond = torch.cat([gated_torque_feat, low_dim_feat], dim=-1)

        # gate the entire feature. have to do it like this because of the precomputation we did
        scale_cond = gate * full_feat + (1 - gate) * self.null_feat

        # dont gate anything for the vision
        vision_unet_cond = torch.cat([vision_feat, low_dim_feat], dim=-1)

        vision_pred = self.vision_unet(
            sample=model_input, timestep=global_timesteps,
            local_cond=local_cond, global_cond=vision_unet_cond,
        )[..., : self.action_dim]

        torque_pred = self.torque_unet(
            sample=model_input, timestep=global_timesteps,
            local_cond=local_cond, global_cond=torque_unet_cond,
        )[..., : self.action_dim]

        pred_scale = F.softplus(self.scale_predictor(scale_cond)) # (B, T)
        pred_scale = pred_scale.view(B, self.horizon, 1)

        # CFG style update
        pred_cfg = vision_pred + pred_scale * (torque_pred - vision_pred)

        target = noise - trajectory

        weights = batch["weights"]  # (B, T)

        # if self.batched_weights: # not supported yet
        #     is_correction = (weights > self.correction_thresh).unsqueeze(-1)  # (B, T, 1)
        #     scale_target = weights  # (B, T)
        # else:  # per_chunk — per-sample trajectory mean, not batch-wide
        #     chunk_weight = weights.mean(dim=1, keepdim=True)                  # (B, 1)
        #     is_correction = (chunk_weight > self.correction_thresh)
        #     is_correction = is_correction.unsqueeze(-1).expand(-1, horizon, -1)  # (B, T, 1)
        #     scale_target = chunk_weight.expand(-1, horizon)                    # (B, T)
        weights = batch["weights"]  # (B, T)
        chunk_weight = weights.mean(dim=1, keepdim=True)  # (B, 1)
        gate_target = (chunk_weight > self.correction_thresh).float()
        scale_target = chunk_weight.expand(-1, horizon)                    # (B, T)

        # i removed loss mask. RTC broken
        cfg_loss = F.mse_loss(pred_cfg, target, reduction="none") * loss_mask.type(pred_cfg.dtype)

        pred_scale = pred_scale.squeeze()
        scale_loss = F.mse_loss(pred_scale, scale_target)

        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_target)
        
        # if self.weighted_loss:
        #     weights = batch["weights"].unsqueeze(-1)
        #     loss = weights * loss
        
        # loss = reduce(loss, "b ... -> b (...)", "mean")
        # loss = loss.mean()
        valid_elements_count = loss_mask.sum().clamp_min(1.0)
        cfg_loss = cfg_loss.sum() / valid_elements_count

        loss = cfg_loss + 0.1 * scale_loss + 0.1 * gate_loss

        loss_dict = {
            "cfg_loss": cfg_loss.item(),
            "scale_loss": scale_loss.item(),
            "gate_loss": gate_loss.item(),
        }

        return loss, loss_dict
