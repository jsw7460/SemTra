from abc import abstractmethod
from typing import List, Dict, Any

import gym


class ComdeSkillEnv(gym.Wrapper):
	def __init__(self, env: gym.Env, seed: int, task: List, n_target: int, cfg: Dict = None):
		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.SKILL_INDEX_MAPPING[task[i]]

		super(ComdeSkillEnv, self).__init__(env=env)

		self.seed = seed
		self.n_target = n_target
		self.cfg = cfg
		"""
			e.g., skill_list: ["door", "stick", "box", "lever"] 
				-> skill_list_idx: [2, 4, 1, 5]  
		"""
		self.skill_idx_list = [self.onehot_skills_mapping[key] for key in self.skill_list]

	@property
	@abstractmethod
	def onehot_skills_mapping(self):
		raise NotImplementedError()

	@property
	def skill_index_mapping(self):
		raise NotImplementedError()

	@abstractmethod
	def get_rtg(self):
		raise NotImplementedError()

	def get_short_str_for_save(self) -> str:
		return "_".join(self.env.skill_list)

	def __str__(self):
		return self.get_short_str_for_save()

	@abstractmethod
	def get_default_parameter(self, non_functionality: str):
		raise NotImplementedError()

	def get_buffer_action(self, action: Any):
		return action