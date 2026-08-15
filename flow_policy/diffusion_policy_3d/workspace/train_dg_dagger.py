if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random
import shutil
import threading
import time

from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.policy.dp3 import DP3
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDgDaggerWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        lastest_ckpt_path = self.get_checkpoint_path()
        
        if getattr(cfg.training, 'resume', False) and lastest_ckpt_path.is_file():
            cprint(f"Resuming training run from local checkpoint {lastest_ckpt_path}", "green")
            self.load_checkpoint(path=lastest_ckpt_path)
        elif getattr(cfg.training, 'sirius_finetune', False):
            # Load the checkpoint but NOT previous optimizer to tune with sirius.
            pretrained_ckpt_path = pathlib.Path(cfg.training.finetune_checkpoint_path)
            self.load_checkpoint(path=pretrained_ckpt_path, include_keys=[], exclude_keys=["optimizer"])

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)

                    grads = [
                        param.grad.detach().flatten()
                        for param in self.model.parameters()
                        if param.grad is not None
                    ]
                    grad_norm = (
                        torch.cat(grads).norm()
                        if len(grads) > 0
                        else 0
                    )

                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'unet_loss': loss_dict["bc_loss"],
                        'base_loss': loss_dict["base_loss"],
                        'corr_loss': loss_dict["corr_loss"],
                        'scale_loss': loss_dict["scale_loss"],
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0],
                        'grad_norm': grad_norm
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()
            
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    # For RTC specifically, condition on some past actions.
                    if hasattr(policy, 'max_delay') and policy.max_delay >= 2:
                        val_delay = torch.randint(1, policy.max_delay + 1, (1,)).item()
                        
                        # Slice the ground truth to create "past actions" to condition inference on
                        # Shape: (B, val_delay, Action_Dim)
                        past_actions_input = gt_action[:, :val_delay, :]
                        result = policy.predict_action(obs_dict, past_actions=past_actions_input)
                        
                        pred_action = result['action_pred']
                        # extra debugging metrics for RTC
                        # Did the model actually overwrite the prefix? Should be ~0.0
                        history_mse = torch.nn.functional.mse_loss(
                            pred_action[:, :val_delay], 
                            past_actions_input
                        )
                        step_log['train_cond_fidelity'] = history_mse.item()

                        # Trying to measure Boundary Smoothness 
                        # We measure the change in velocity across the boundary (indices: d-2, d-1, d)
                        # v_in: Velocity of the last step of history
                        v_in = pred_action[:, val_delay-1] - pred_action[:, val_delay-2]
                        
                        # v_out: Velocity of the first predicted step
                        v_out = pred_action[:, val_delay] - pred_action[:, val_delay-1]

                        boundary_jerk = torch.nn.functional.mse_loss(v_out, v_in)
                        step_log['train_boundary_smoothness'] = boundary_jerk.item()
                        
                    else:
                        val_delay = 0
                        result = policy.predict_action(obs_dict)
                        
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    
                    
                    # TODO: add a config flag on whether we do this plotting,
                    # since I have a feeling it might be slow..
                    gt_np = gt_action[0].detach().cpu().numpy()
                    pred_np = pred_action[0].detach().cpu().numpy()
                    
                    # 2. Setup Plot (Limit to 6 dimensions to keep it readable)
                    n_dims = gt_np.shape[-1]
                    n_plots = min(n_dims, 6)
                    
                    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
                    if n_plots == 1: axes = [axes] # Handle single dim case

                    for d in range(n_plots):
                        ax = axes[d]
                        # Plot Ground Truth (Black Solid)
                        ax.plot(gt_np[:, d], label='Ground Truth', color='black', alpha=0.6, linewidth=2)
                        # Plot Prediction (Red Dashed)
                        ax.plot(pred_np[:, d], label='Prediction', color='red', linestyle='--', alpha=0.8, linewidth=2)
                        
                        # 3. Highlight the "History" Context if used
                        if val_delay > 0:
                            ax.axvline(x=val_delay - 1, color='blue', linestyle=':', alpha=0.5)
                            ax.axvspan(0, val_delay - 1, color='blue', alpha=0.1, label='Conditioning (History)')
                            
                        ax.set_ylabel(f'Dim {d}')
                        ax.grid(True, alpha=0.3)
                        if d == 0:
                            ax.legend(loc='upper right', fontsize='small')

                    plt.tight_layout()
                    
                    # 4. Add Image to step_log
                    step_log['train_trajectory_vis'] = wandb.Image(fig)
                    
                    plt.close(fig)
                    
                    del fig
                    del axes
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            # if env_runner is None:
            step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        ckpt_path = self.get_checkpoint_path(tag=cfg.checkpoint.tag)
        if ckpt_path.is_file():
            cprint(f"EVAL on checkpoint {ckpt_path}", 'magenta')
            self.load_checkpoint(path=ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDgDaggerWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
