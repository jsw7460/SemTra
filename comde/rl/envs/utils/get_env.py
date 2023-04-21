import pickle
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Type


import gym

from comde.rl.envs.metaworld import MultiStageMetaWorld
from comde.rl.envs.franka_kitchen import FrankaKitchen
from comde.rl.envs.utils import TimeLimitEnv, SkillHistoryEnv, SkillInfoEnv
import random


def get_dummy_env(cfg: Dict) -> SkillInfoEnv:
	"""
	Note: This is not responsible for evaluate the env.
	"""
	env_name = cfg["name"].lower()

	if "metaworld" in env_name:
		# Task has no meaning here
		env = MultiStageMetaWorld(seed=0, task=["box", "handle", "button", "door"], n_target=4)

	elif "kitchen" in env_name:
		# Task has no meaning here
		env = FrankaKitchen(seed=0, task=["microwave", "kettle", "top burner", "hinge cabinet"], n_target=4)

	else:
		raise NotImplementedError(f"Not supported: {env_name}")

	with open(Path(cfg["skill_infos_path"]), "rb") as f:
		skill_infos = pickle.load(f)

	env = SkillInfoEnv(env=env, skill_infos=skill_infos)
	return env


def get_batch_env(
	env_class: Type[gym.Env],
	tasks: List,
	skill_dim: int,
	time_limit: int = 1000,
	history_len: int = 1,
	seed: int = 0,
	cfg: Dict = None,
) -> List[SkillHistoryEnv]:
	envs = []	# type: List[SkillHistoryEnv]

	with open(Path(cfg["skill_infos_path"]), "rb") as f:
		skill_infos = pickle.load(f)

	# tasks = list(permutations([1, 3, 4, 6], r=4))
	# for i in range(len(tasks)):
	# 	tasks[i] = list(tasks[i])
	#
	# random.shuffle(tasks)

	for task in tasks:
		env = env_class(seed=seed, task=task, n_target=3, cfg=cfg)
		env = SkillInfoEnv(env, skill_infos=skill_infos)
		env.availability_check()
		env = TimeLimitEnv(env=env, limit=time_limit)
		env = SkillHistoryEnv(env=env, skill_dim=skill_dim, num_stack_frames=history_len)
		envs.append(env)
	return envs
