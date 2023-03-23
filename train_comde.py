import random

random.seed(0)

import os
from os.path import join, isfile
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from comde.trainer.comde_trainer import ComdeTrainer
from comde.rl.buffers import ComdeBuffer
from comde.rl.envs import get_dummy_env


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)

	env = get_dummy_env(cfg["env"])	# Dummy env for obtain an observation and action space.
	low_policy = instantiate(cfg["low_policy"])

	"""
	low_policy = SkillDecisionTransformer(cfg)
	"""
	seq2seq = instantiate(cfg["seq2seq"])
	termination = instantiate(cfg["termination"])

	trainer = ComdeTrainer(
		cfg=cfg,
		low_policy=low_policy,
		seq2seq=seq2seq,
		termination=termination,
		skill_to_vec=env.skill_to_vec
	)

	data_dirs = [
		Path(cfg["dataset_path"]) / Path(name) for name in os.listdir(cfg["dataset_path"])
	]

	for n_iter in range(cfg["max_iter"]):
		n_iter = (n_iter % len(data_dirs))
		data_dir = data_dirs[n_iter]
		hdf_files = [join(data_dir, f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
		random.shuffle(hdf_files)

		replay_buffer = ComdeBuffer(
			observation_space=env.observation_space,
			action_space=env.action_space,
			subseq_len=cfg["subseq_len"]
		)
		replay_buffer.add_episodes_from_h5py({
			"trajectory": hdf_files[: -1],
			"language_guidance": cfg["language_guidance_path"]
		})
		trainer.run(replay_buffer)

		eval_buffer = ComdeBuffer(
			observation_space=env.observation_space,
			action_space=env.action_space,
			subseq_len=cfg["subseq_len"]
		)
		eval_buffer.add_episodes_from_h5py({
			"trajectory": hdf_files[-1:],
			"language_guidance": cfg["language_guidance_path"]
		})
		trainer.evaluate(eval_buffer)
		trainer.save()


if __name__ == "__main__":
	program()
