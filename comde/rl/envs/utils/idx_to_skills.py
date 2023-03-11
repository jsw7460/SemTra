import gym

from typing import Dict


class IdxToSkill(gym.Wrapper):
	def __init__(self, env: gym.Env, idx_to_skills: Dict):
		super(IdxToSkill, self).__init__(env=env)
		self.__idx_to_skills = idx_to_skills

	@property
	def idx_to_skills(self):
		return self.__idx_to_skills
