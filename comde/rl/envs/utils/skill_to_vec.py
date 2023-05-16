from typing import Dict, List, Union
import random

import gym
import numpy as np

from comde.utils.common.lang_representation import SkillRepresentation


class SkillInfoEnv(gym.Wrapper):
	def __init__(self, env: gym.Env, skill_infos: Dict[str, List[SkillRepresentation]]):
		super(SkillInfoEnv, self).__init__(env=env)

		assert hasattr(env, "onehot_skills_mapping"), \
			"Environment should have skill mapping: skill(e.g., `drawer`) -> index (e.g., `3`)"

		assert hasattr(env, "skill_list"), \
			"Environment should have skill list (e.g., [`drawer`, `button`, `box`])"

		self.__skill_infos = skill_infos
		self.__skill_to_vec = None
		self.__idx_skill_list = []

		skills = list(self.__skill_infos.values())

		# NOTE: This entitle onehot representation of each vector
		skills.sort(key=lambda skill: skill[0].index)

		for sk in self.skill_list:
			self.__idx_skill_list.append(self.onehot_skills_mapping[sk])

	@property
	def skill_infos(self):
		return self.__skill_infos

	@property
	def idx_skill_list(self):
		return self.__idx_skill_list

	def availability_check(self):
		# Check whether onehot index and loaded information match well.
		skills = []
		for sk in self.skill_infos.values():
			skills.extend(sk)

		for sk in skills:
			assert sk.index == self.env.onehot_skills_mapping[sk.title], \
				f"Skill onehot representation mismatch."

	def get_skill_from_idx(self, idx: int, variation: Union[str, int] = "random") -> SkillRepresentation:
		sk = self.skill_index_mapping[idx]
		if variation == "random":
			return random.choice(self.skill_infos[sk])
		else:
			return self.skill_infos[sk][variation]

	def get_skill_vectors_from_idx_list(self, idxs: List[int]) -> np.ndarray:
		skill_vectors = []
		for idx in idxs:
			skill = self.get_skill_from_idx(idx)
			skill_vectors.append(skill.vec)
		skill_vectors = np.array(skill_vectors)
		return skill_vectors