import pickle
from pathlib import Path
from typing import Dict, List, Type, Union

import gym

from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.utils import TimeLimitEnv, SkillHistoryEnv, SkillInfoEnv


def get_dummy_env(env_name: str, cfg: Dict = None) -> SkillInfoEnv:
	"""
	Note: This is not responsible for the evaluation of env.
	"""

	skill_infos = None
	# Task has no meaning here
	if "metaworld" in env_name:
		from comde.rl.envs.metaworld import MultiStageMetaWorld, metaworld_skill_infos
		env = MultiStageMetaWorld(seed=0, task=["box", "handle", "button", "door"], n_target=4)
		skill_infos = metaworld_skill_infos
	elif "kitchen" in env_name:
		from comde.rl.envs.franka_kitchen import FrankaKitchen, kitchen_skill_infos
		env = FrankaKitchen(seed=0, task=["microwave", "kettle", "bottom burner", "top burner"], n_target=4)
		skill_infos = kitchen_skill_infos
	elif "carla" in env_name:
		from comde.rl.envs.carla import DummyEasyCarla
		env = DummyEasyCarla(seed=0, task=["straight", "right", "left"], n_target=3, cfg=cfg)
	elif "rlbench" in env_name:
		from comde.rl.envs.rlbench import RLBench, rlbench_skill_infos
		env = RLBench(seed=0, task=[0, 3, 6, 9], n_target=4, dummy=True, cfg=cfg)
		skill_infos = rlbench_skill_infos
	else:
		raise NotImplementedError(f"Not supported: {env_name}")
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
	cfg: Dict = None
):
	env = env_class(seed=seed, task=task, n_target=n_target, cfg=cfg)
	skill_infos = env_class.get_skill_infos()
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
) -> List[Union[SkillHistoryEnv, ComdeSkillEnv]]:
	envs = []  # type: List[SkillHistoryEnv]

	# with open(Path(cfg["skill_infos_path"]), "rb") as f:
	# 	skill_infos = pickle.load(f)

	for task in tasks:
		env = get_env(
			env_class=env_class,
			task=task,
			n_target=n_target,
			skill_dim=skill_dim,
			time_limit=time_limit,
			history_len=history_len,
			seed=seed,
			cfg=cfg
		)
		envs.append(env)
	return envs
