from typing import List, Dict

import gym
import numpy as np

from meta_world.get_video import SingleTask
from comde.rl.envs.base import ComdeSkillEnv


class MultiStageMetaWorld(ComdeSkillEnv):
	MW_OBS_DIM = 140
	onehot_skills_mapping = {
		"box": 0,
		"puck": 1,
		"handle": 2,
		"drawer": 3,
		"button": 4,
		"lever": 5,
		"door": 6,
		"stick": 7
	}
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		base_env = SingleTask(seed=seed, task=task)
		super(MultiStageMetaWorld, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, MultiStageMetaWorld.MW_OBS_DIM))

	def step(self, action):
		obs, reward, done, info = super(MultiStageMetaWorld, self).step(action)
		if self.env.env.mode == self.n_target:
			done = True
		return obs, reward, done, info
