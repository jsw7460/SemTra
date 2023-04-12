import pickle
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Type

import gym

from comde.rl.envs.metaworld.dimfix import DimFixedMetaWorld
from comde.rl.envs.utils import TimeLimitEnv, SkillHistoryEnv, SkillInfoEnv


def get_dummy_env(cfg: Dict) -> SkillInfoEnv:
	"""
	Note: This is not responsible for evaluate the env.
	"""
	env_name = cfg["name"]

	if "meta" in env_name.lower():
		env = DimFixedMetaWorld(seed=0, task=["box", "handle", "button", "door"])  # No meaning
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
	# with open("/home/jsw7460/mnt/comde_datasets/language_embeddings/clip_mappings/metaworld/clip_mapping", "rb") as f:
		skill_infos = pickle.load(f)

	tasks = list(permutations([1, 3, 4, 6], r=4))
	for i in range(len(tasks)):
		tasks[i] = list(tasks[i])

	for task in tasks:
		env = env_class(seed=seed, task=task, cfg=cfg)
		env = SkillInfoEnv(env, skill_infos=skill_infos)
		env.availability_check()
		env = TimeLimitEnv(env=env, limit=time_limit)
		env = SkillHistoryEnv(env=env, skill_dim=skill_dim, num_stack_frames=history_len)
		envs.append(env)
	return envs
