import pickle
from pathlib import Path
from typing import Dict, List, Type

import gym

from comde.rl.envs.carla import DummyEasyCarla
from comde.rl.envs.franka_kitchen import FrankaKitchen
from comde.rl.envs.metaworld import MultiStageMetaWorld
from comde.rl.envs.utils import TimeLimitEnv, SkillHistoryEnv, SkillInfoEnv


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
		env = FrankaKitchen(seed=0, task=["microwave", "kettle", "bottom burner", "top burner"], n_target=4)

	elif "carla" in env_name:
		env = DummyEasyCarla(seed=0, task=["straight", "right", "left"], n_target=3, cfg=cfg)

	else:
		raise NotImplementedError(f"Not supported: {env_name}")

	with open(Path(cfg["skill_infos_path"]), "rb") as f:
		skill_infos = pickle.load(f)

	env = SkillInfoEnv(env=env, skill_infos=skill_infos)
	return env


def get_env(
	env_class: Type[gym.Env],
	task: List,
	n_target: int,
	skill_dim: int,
	time_limit: int = 1000,
	history_len: int = 1,
	seed: int = 0,
	skill_infos: Dict = None,
	cfg: Dict = None
):
	if skill_infos is None:
		with open(Path(cfg["skill_infos_path"]), "rb") as f:
			skill_infos = pickle.load(f)

	env = env_class(seed=seed, task=task, n_target=n_target, cfg=cfg)
	env = SkillInfoEnv(env, skill_infos=skill_infos)
	env.availability_check()
	env = TimeLimitEnv(env=env, limit=time_limit)
	env = SkillHistoryEnv(env=env, skill_dim=skill_dim, num_stack_frames=history_len)
	return env

def get_batch_env(
	env_class: Type[gym.Env],
	tasks: List,
	n_target: int,
	skill_dim: int,
	time_limit: int = 1000,
	history_len: int = 1,
	seed: int = 0,
	cfg: Dict = None,
) -> List[SkillHistoryEnv]:
	envs = []  # type: List[SkillHistoryEnv]

	with open(Path(cfg["skill_infos_path"]), "rb") as f:
		skill_infos = pickle.load(f)

	for task in tasks:
		env = get_env(
			env_class=env_class,
			task=task,
			n_target=n_target,
			skill_dim=skill_dim,
			time_limit=time_limit,
			history_len=history_len,
			skill_infos=skill_infos,
			seed=seed,
			cfg=cfg
		)
		envs.append(env)
	return envs
