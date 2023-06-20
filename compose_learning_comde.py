import os
import random
from os.path import join, isfile
from pathlib import Path
from typing import Dict
from typing import Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from comde.rl.envs import get_dummy_env
from comde.trainer.compose_trainer import ComposeTrainer
from comde.utils.common.natural_languages.merge_tokens import merge_env_tokens


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict[str, Union[str, int, Dict]]

	assert cfg["mode"]["mode"] == "compose_learning", \
		f"Your mode is {cfg['mode']['mode']}. " \
		"Please add 'mode=compose_learning' to your command line if you want to train composition module training"

	metaworld = get_dummy_env("metaworld")
	kitchen = get_dummy_env("kitchen")
	rlbench = get_dummy_env("rlbench")

	envs = {"metaworld": metaworld, "kitchen": kitchen, "rlbench": rlbench}
	tokens, offset_info = merge_env_tokens(list(envs.values()))

	env_datasets = [
		"/home/jsw7460/mnt/comde_datasets/metaworld/speed/0508/3_target_skills/",  # Meta world
		"/home/jsw7460/mnt/comde_datasets/kitchen/wind/4_target_skills/",  # Kitchen
		"/home/jsw7460/mnt/comde_datasets/rlbench/weight/3_target_skills/",
		"/home/jsw7460/mnt/comde_datasets/rlbench/weight/4_target_skills/"
	]

	data_dirs = []
	hdf_files = []
	for env_data in env_datasets:
		_data_dirs = [Path(env_data) / Path(name) for name in os.listdir(env_data) if name not in cfg["excluded_dirs"]]
		data_dirs.extend(_data_dirs)
		for data_dir in _data_dirs:
			hdf_files.extend([join(data_dir, f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))])

	random.shuffle(hdf_files)
	dataset_window_size = len(hdf_files) // len(data_dirs)

	seq2seq = instantiate(cfg["compose_learner"], custom_tokens=tokens)
	trainer = ComposeTrainer(cfg=cfg, envs=envs, offset_info=offset_info, seq2seq=seq2seq)

	for n_iter in range(cfg["max_iter"]):
		n_iter = (n_iter % len(data_dirs))
		trajectories = hdf_files[n_iter * dataset_window_size: (n_iter + 1) * dataset_window_size]
		trainer.run(trajectories=trajectories)

		eval_trajectories = random.choices(hdf_files, k=50)
		trainer.evaluate(eval_trajectories)

if __name__ == "__main__":
	program()
