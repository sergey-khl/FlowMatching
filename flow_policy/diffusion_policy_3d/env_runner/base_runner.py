from typing import Dict, Optional
from diffusion_policy_3d.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy, res_policy: Optional[BasePolicy] = None) -> Dict:
        raise NotImplementedError()
