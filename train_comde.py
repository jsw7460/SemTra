from jax.config import config

config.update("jax_debug_nans", True)

from typing import Dict
import random

random.seed(7)

import os
from os.path import join, isfile
from pathlib import Path

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

from comde.rl.buffers import ComdeBuffer
from comde.rl.envs import get_dummy_env
from comde.utils.common.normalization import get_observation_statistics


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)

	data_dirs = [Path(cfg["dataset_path"]) / Path(name) for name in os.listdir(cfg["dataset_path"])]
	hdf_files = []
	for data_dir in data_dirs:
		hdf_files.extend([join(data_dir, f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))])
	random.shuffle(hdf_files)
	assert len(hdf_files) > 0, "Empty dataset"

	dataset_window_size = len(hdf_files) // len(data_dirs)

	if cfg["state_normalization"]:
		statistics = get_observation_statistics(data_dirs)
		low_policy_cfgs = cfg["low_policy"]  # type: Dict
		low_policy_cfgs["cfg"].update({**statistics})

	env = get_dummy_env(cfg["env"])  # Dummy env for obtain an observation and action space.
	modules_dict = {module: instantiate(cfg[module]) for module in cfg["modules"]}

	trainer_cls = get_class(cfg["trainer"])

	trainer = trainer_cls(
		cfg=cfg,
		skill_infos=env.skill_infos,
		**modules_dict
	)

	for n_iter in range(cfg["max_iter"]):
		n_iter = (n_iter % len(data_dirs))
		trajectories = hdf_files[n_iter * dataset_window_size: (n_iter + 1) * dataset_window_size]

		replay_buffer = ComdeBuffer(
			observation_space=env.observation_space,
			action_space=env.action_space,
			subseq_len=cfg["subseq_len"]
		)
		replay_buffer.add_episodes_from_h5py(
			paths={
				"trajectory": trajectories[: -10],
				"sequential_requirements": cfg["sequential_requirements_path"],
				"non_functionalities": cfg["non_functionalities_path"]
			},
			cfg=cfg["dataset"],
			mode="train"
		)

		trainer.run(replay_buffer)

		eval_buffer = ComdeBuffer(
			observation_space=env.observation_space,
			action_space=env.action_space,
			subseq_len=cfg["subseq_len"]
		)
		eval_buffer.add_episodes_from_h5py(
			paths={
				"trajectory": trajectories[-10:],
				"sequential_requirements": cfg["sequential_requirements_path"],
				"non_functionalities": cfg["non_functionalities_path"]
			},
			cfg=cfg["dataset"],
			mode="eval"
		)
		trainer.evaluate(eval_buffer)
		trainer.save()


if __name__ == "__main__":
	program()
