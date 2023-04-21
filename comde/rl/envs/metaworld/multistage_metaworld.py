from typing import List, Dict

import gym
import numpy as np

from meta_world.get_video import SingleTask

MW_OBS_DIM = 140


class MultiStageMetaWorld(gym.Wrapper):
	ONEHOT_SKILLS_MAPPING = {
		"box": 0,
		"puck": 1,
		"handle": 2,
		"drawer": 3,
		"button": 4,
		"lever": 5,
		"door": 6,
		"stick": 7
	}

	SKILL_INDEX_MAPPING = {v: k for k, v in ONEHOT_SKILLS_MAPPING.items()}

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.SKILL_INDEX_MAPPING[task[i]]

		base_env = SingleTask(seed=seed, task=task)
		super(MultiStageMetaWorld, self).__init__(env=base_env)

		"""
			e.g., skill_list: ["door", "stick", "box", "lever"] 
				-> skill_list_idx: [2, 4, 1, 5]  
		"""
		self.n_target = n_target
		self.cfg = cfg
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, MW_OBS_DIM))
		self.skill_idx_list = [self.ONEHOT_SKILLS_MAPPING[key] for key in self.skill_list]

	def get_short_str_for_save(self) -> str:
		return "_".join(self.env.skill_list)

	def __str__(self):
		return self.get_short_str_for_save()

	def step(self, action):
		obs, reward, done, info = super(MultiStageMetaWorld, self).step(action)
		if self.env.env.mode == self.n_target:
			done = True
		return obs, reward, done, info