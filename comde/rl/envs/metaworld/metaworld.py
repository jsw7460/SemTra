from typing import List

import gym
import numpy as np

from meta_world.get_video import SingleTask

MW_OBS_DIM = 140


class DimFixedMetaWorld(gym.Wrapper):
	def __init__(self, seed: int, task: List):
		base_env = SingleTask(seed=seed, task=task)
		super(DimFixedMetaWorld, self).__init__(env=base_env)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, MW_OBS_DIM))

	def get_short_str_for_save(self) -> str:
		return "".join([skill[0] for skill in self.env.skill_list])
