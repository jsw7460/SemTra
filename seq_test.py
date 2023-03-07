import warnings

warnings.simplefilter("ignore", UserWarning)
import logging

logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)

import os
from os.path import join, isfile
from pathlib import Path

import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from comde.trainer.comde_trainer import ComdeTrainer
from comde.comde_modules.low_policies import SkillDecisionTransformer
from comde.comde_modules.seq2seq import SkillToSkillLSTM
from comde.comde_modules.termination import MLPTermination
from comde.rl.buffers import ComdeBuffer
from comde.rl.envs import get_dummy_env


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	seed = cfg["seed"]

	cfg.env = "metaworld"
	cfg.skill_dim = 512
	cfg.act_scale = 1.0
	cfg.observation_dim = 140
	cfg.action_dim = 4

	cfg = OmegaConf.to_container(cfg, resolve=True)

	env = get_dummy_env(cfg["env"])

	low_policy = SkillDecisionTransformer(seed=seed, cfg=cfg["low_policy"].copy())
	seq2seq = SkillToSkillLSTM(seed=seed, cfg=cfg["seq2seq"].copy())
	termination = MLPTermination(seed=seed, cfg=cfg["termination"].copy())

	replay_buffer = ComdeBuffer(
		observation_space=env.observation_space,
		action_space=env.action_space,
		subseq_len=cfg["subseq_len"]
	)

	idx_to_skill = {f"{i}": np.zeros((512,)) for i in range(8)}


	trainer = ComdeTrainer(
		cfg=cfg,
		low_policy=low_policy,
		seq2seq=seq2seq,
		termination=termination,
		idx_to_skill=idx_to_skill
	)

	data_dirs = [
		Path(cfg["dataset_path"]) / Path(name) for name in os.listdir(cfg["dataset_path"])
	]

	for n_iter in range(cfg["max_iter"]):
		n_iter = (n_iter % len(data_dirs))
		data_dir = data_dirs[n_iter]
		hdf_files = [join(data_dir, f) for f in os.listdir(data_dir) if isfile(join(data_dir, f))]

		replay_buffer.add_episodes_from_h5py(hdf_files)
		trainer.replay_buffer = replay_buffer	# Set replay buffer

		trainer.run()

if __name__ == "__main__":
	program()
