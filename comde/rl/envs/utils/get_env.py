import gym
from meta_world.get_video import SingleTask
from comde.rl.envs.utils.dimfix import DimensionFix
from comde.rl.envs.metaworld.metaworld import (
	observation_space as mw_observation_space,
	action_space as mw_action_space
)


def get_dummy_env(env_name: str) -> gym.Env:
	"""
	Note: This is not responsible for evaluate the env.
	"""
	if "meta" in env_name.lower():
		env = SingleTask(seed=0, skill_list=["box", "handle", "button", "door"])		# No meaning
		env = DimensionFix(env=env, observation_space=mw_observation_space, action_space=mw_action_space)

	else:
		raise NotImplementedError(f"Not supported: {env_name}")

	return env
