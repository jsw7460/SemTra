# from jax.config import config
#
# config.update("jax_debug_nans", True)

import random
from typing import Dict, Union

random.seed(7)

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from comde.trainer.hrl_trainer import HierarchicalRLAgent

from comde.rl.envs import get_env


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict[str, Union[str, int, Dict]]

	modules_dict = {module: instantiate(cfg[module]) for module in cfg["modules"]}
	pretrained_modules = HierarchicalRLAgent.load_pretrained_modules(cfg=cfg)
	low_policy = pretrained_modules["low_policy"]
	modules_dict.update({**pretrained_modules})

	skill_dim = low_policy.cfg["skill_dim"]
	non_functionality_dim = low_policy.cfg["non_functionality_dim"]
	param_dim = low_policy.cfg["param_dim"]
	param_repeats = low_policy.cfg["param_repeats"]
	total_param_dim = param_dim * param_repeats
	subseq_len = low_policy.cfg.get("subseq_len", 1)

	env_class = get_class(cfg["env"]["path"])
	env = get_env(
		env_class=env_class,
		task=cfg["mode"]["tasks"],
		cfg=cfg,
		parameterized_skill_dim=skill_dim + non_functionality_dim + total_param_dim,
		time_limit=cfg["env"]["timelimit"],
		history_len=subseq_len,
		seed=cfg["seed"]
	)  # Dummy env for obtain an observation and action space.

	trainer = HierarchicalRLAgent(
		cfg=cfg,
		env=env,
		skill_infos=env.skill_infos,
		**modules_dict
	)
	trainer.run()


if __name__ == "__main__":
	program()
