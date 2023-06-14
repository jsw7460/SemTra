from jax.config import config

config.update("jax_debug_nans", True)

import random
from typing import Dict, Union

random.seed(7)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from comde.rl.envs.metaworld.multistage_metaworld import MultiStageMetaWorld
from comde.rl.envs.franka_kitchen.franka_kitchen import FrankaKitchen
from comde.rl.envs.rlbench.rlbench import RLBench
from comde.trainer.prompt_trainer import PromptTrainer


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict[str, Union[str, int, Dict]]

	assert cfg["mode"]["mode"] == "prompt_learning", \
		f"Your mode is {cfg['mode']}. " \
		"Please add 'mode=prompt_learning' to your command line if you want to train prompt learning"

	prompt_cfg = cfg["mode"]

	envs = [MultiStageMetaWorld, FrankaKitchen, RLBench]
	prompt_learner = instantiate(cfg["prompt_learner"])
	prompt_trainer = PromptTrainer(cfg=cfg, envs=envs, prompt_learner=prompt_learner)

	for _ in range(prompt_cfg["max_iter"]):
		prompt_trainer.run()


if __name__ == "__main__":
	program()
