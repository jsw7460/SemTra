import gym

from typing import Dict


class SkillToVec(gym.Wrapper):
	def __init__(self, env: gym.Env, skill_to_vec: Dict):
		super(SkillToVec, self).__init__(env=env)
		self.__skill_to_vec = skill_to_vec

	@property
	def skill_to_vec(self):
		return self.__skill_to_vec
