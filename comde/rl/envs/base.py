from abc import abstractmethod
from typing import List, Dict, Any, Union

import gym
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward


class ComdeSkillEnv(gym.Wrapper):
	def __init__(self, env: gym.Env, seed: int, task: List, n_target: int, cfg: Dict = None):

		super(ComdeSkillEnv, self).__init__(env=env)

		self.seed = seed
		self.n_target = n_target
		self.cfg = cfg
		self.task = task
		"""
			e.g., skill_list: ["door", "stick", "box", "lever"] 
				-> skill_list_idx: [2, 4, 1, 5]  
		"""
		self.skill_idx_list = [self.onehot_skills_mapping[key] for key in self.skill_list]
		self.sequential_requirements_vector_mapping = None
		self.non_functionalities_vector_mapping = None

	def set_sequential_requirements_mapping(self, sequential_requirements: Dict):
		self.sequential_requirements_vector_mapping = {
			seq_req: dict() for seq_req in sequential_requirements.keys()
		}
		for seq_req, variations in sequential_requirements.items():
			for variation in variations:
				vec = bert_base_forward(variation)["language_embedding"][:, 0, ...]
				vec = vec.reshape(-1, )
				self.sequential_requirements_vector_mapping[seq_req][variation] = vec

	def set_non_functionalities_mapping(self, non_functionalities: Dict):
		self.non_functionalities_vector_mapping = {
			seq_req: dict() for seq_req in non_functionalities.keys()
		}
		for seq_req, variations in non_functionalities.items():
			for variation in variations:
				vec = bert_base_forward(variation)["language_embedding"][:, 0, ...]
				vec = vec.reshape(-1, )
				self.non_functionalities_vector_mapping[seq_req][variation] = vec

	@staticmethod
	def ingradients_to_target(non_functionality: str, skill: str, param: str):
		return " ".join([non_functionality, skill, param])

	@staticmethod
	def target_to_ingradients(target: str):
		target = target.split(" ")
		non_functionality = target[0]
		skill = target[1]
		param = "_".join(target[2:])
		if len(param) == 0:
			param = "one"
		return {"non_functionality": non_functionality, "skill": skill, "param": param}

	@abstractmethod
	def ingradients_to_parameter(self, prompt_extraction: str):
		raise NotImplementedError()

	@property
	@abstractmethod
	def onehot_skills_mapping(self):
		raise NotImplementedError()

	@staticmethod
	def eval_param(self):
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

	@staticmethod
	def get_default_parameter(non_functionality: str):
		raise NotImplementedError()

	def get_buffer_action(self, action: Any):
		return action

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		parameter: Union[float, Dict],
		source_skills_idx: List[int],
		video_parsing: bool
	):
		raise NotImplementedError()

	@staticmethod
	def idxs_to_str_skills(skill_index_mapping: Dict, idxs: List[int]) -> List[str]:
		return [skill_index_mapping[sk] for sk in idxs]

	@staticmethod
	def replace_idx_so_skill(skill_index_mapping: Dict, sentence: str) -> str:
		sentence = sentence.split()
		for t, word in enumerate(sentence):
			if word.isdigit():
				sentence[t] = skill_index_mapping[int(word)]

		return " ".join(sentence)

	@staticmethod
	def generate_random_language_guidance():
		raise NotImplementedError()