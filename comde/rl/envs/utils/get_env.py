import pickle
from pathlib import Path
from typing import Dict, List, Type

import gym

from comde.rl.envs.metaworld.dimfix import DimFixedMetaWorld
from comde.rl.envs.utils import TimeLimitEnv, SkillHistoryEnv
from comde.rl.envs.utils.skill_to_vec import SkillToVec


def get_dummy_env(cfg: Dict) -> SkillToVec:
	"""
	Note: This is not responsible for evaluate the env.
	"""
	env_name = cfg["name"]

	if "meta" in env_name.lower():
		env = DimFixedMetaWorld(seed=0, task=["box", "handle", "button", "door"])  # No meaning
	else:
		raise NotImplementedError(f"Not supported: {env_name}")

	with open(Path(cfg["skill_to_vec_path"]), "rb") as f:
		skill_to_vec = pickle.load(f)

	env = SkillToVec(env=env, skill_to_vec=skill_to_vec)

	return env


def get_batch_env(
	env_class: Type[gym.Env],
	tasks: List,
	skill_dim: int,
	time_limit: int = 1000,
	history_len: int = 1,
	seed: int = 0
) -> List[SkillHistoryEnv]:
	envs = []
	for task in tasks:
		env = env_class(seed=seed, task=task)  # type: gym.Env
		env = TimeLimitEnv(env=env, limit=time_limit)  # type: gym.Env
		env = SkillHistoryEnv(env=env, skill_dim=skill_dim, num_stack_frames=history_len)
		envs.append(env)
	return envs
