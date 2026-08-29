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

class EvalDgDaggerWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        self.cfg = cfg
        self._output_dir = cfg.output_dir
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
    workspace = EvalDgDaggerWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
