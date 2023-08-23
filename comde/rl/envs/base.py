from abc import abstractmethod
import pickle
from typing import List, Dict, Any, Union, Tuple

from copy import deepcopy

import gym
import numpy as np
from comde.utils.common import pretrained_forwards


SEQUENTIAL_REQUIREMENTS = None
NON_FUNCTIONALITIES = None

# with open("/home/jsw7460/metaworld_sequential_requirements_mapping", "rb") as f:
# 	SEQUENTIAL_REQUIREMENTS = pickle.load(f)
#
# with open("/home/jsw7460/metaworld_non_functionalities_mapping", "rb") as f:
# 	NON_FUNCTIONALITIES = pickle.load(f)


class ComdeSkillEnv(gym.Wrapper):

	def __init__(self, env: gym.Env, seed: int, task: List, n_target: int, cfg: Dict = None):

		super(ComdeSkillEnv, self).__init__(env=env)
		self.language_space = cfg.get("language_space", "bert")
		self._language_encoder_forward = getattr(pretrained_forwards, self.language_space)
		self.seed = seed
		self.n_target = n_target
		self.cfg = cfg
		self.task = task
		"""
			e.g., skill_list: ["door", "stick", "box", "lever"] 
				-> skill_list_idx: [2, 4, 1, 5]  
		"""
		self.skill_idx_list = [self.onehot_skills_mapping[key] for key in self.skill_list]
		self.str_parameter = None	# type: str

	def get_idx_from_parameter(self, *args, **kwargs):
		raise NotImplementedError()

	def get_parameter_from_idx(self, *args, **kwargs):
		raise NotImplementedError()

	def get_idx_to_parameter_dict(self):
		raise NotImplementedError()

	def set_str_parameter(self, parameter: str):
		self.str_parameter = parameter

	def get_sequential_requirements_mapping(self, sequential_requirements: Dict):
		if SEQUENTIAL_REQUIREMENTS is not None:
			return SEQUENTIAL_REQUIREMENTS
		else:
			mapping = {
				seq_req: dict() for seq_req in sequential_requirements.keys()
			}
			for seq_req, variations in sequential_requirements.items():
				for variation in variations:
					vec = self._language_encoder_forward(variation)["language_embedding"]
					_vec = vec[:, 0, ...]
					del vec
					_vec = _vec.reshape(-1, )
					mapping[seq_req][variation] = _vec

			# with open("/home/jsw7460/metaworld_sequential_requirements_mapping", "wb") as f:
			# 	pickle.dump(mapping, f)
			# print("Save seqreq !!!!!!!!!!!!!!!!!!!" * 999)
			return mapping

	def get_non_functionalities_mapping(self, non_functionalities: Dict):
		if NON_FUNCTIONALITIES is not None:
			return NON_FUNCTIONALITIES
		else:
			mapping = {
				seq_req: dict() for seq_req in non_functionalities.keys()
			}
			for seq_req, variations in non_functionalities.items():
				for variation in variations:
					vec = self._language_encoder_forward(variation)["language_embedding"][:, 0, ...]
					vec = vec.reshape(-1, )
					mapping[seq_req][variation] = vec

			# with open("/home/jsw7460/metaworld_non_functionalities_mapping", "wb") as f:
			# 	pickle.dump(mapping, f)
			# print("Save nf !!!!!!!!!!!!!!!!!!!" * 999)
			return mapping

	@staticmethod
	@abstractmethod
	def get_skill_infos():
		raise NotImplementedError

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
	def get_default_parameter(non_functionality: Union[str, None] = None):
		raise NotImplementedError()

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None,
		video_parsing: bool = True
	):
		raise NotImplementedError()

	@staticmethod
	def idxs_to_str_skills(skill_index_mapping: Dict, idxs: Union[List[int], Tuple[int]]) -> List[str]:
		return [skill_index_mapping[sk] for sk in idxs]

	@staticmethod
	def str_to_idxs_skills(
		onehot_skills_mapping: Dict,
		skills: Union[List[str], Tuple[str]],
		to_str: bool = False
	) -> Union[List[int], List[str]]:
		if to_str:
			return [str(onehot_skills_mapping[sk]) for sk in skills]
		else:
			return [onehot_skills_mapping[sk] for sk in skills]

	@staticmethod
	def replace_idx_so_skill(skill_index_mapping: Dict, sentence: str) -> str:
		sentence = sentence.split()
		for t, word in enumerate(sentence):
			if word.isdigit():
				sentence[t] = skill_index_mapping[int(word)]

		return " ".join(sentence)

	@staticmethod
	def get_target_skill_from_source(
		source_skills_idx: List[int],
		sequential_requirement: str,
		avoid_impossible: bool = False
	):
		if sequential_requirement == "sequential":
			target_skills_idx = source_skills_idx
		elif sequential_requirement == "reverse":
			target_skills_idx = list(reversed(source_skills_idx))
		else:
			changed_idxs = []
			for _str in sequential_requirement:
				if _str.isdigit():
					changed_idxs.append(eval(_str))
			_from = changed_idxs[0]	# 4
			_to = changed_idxs[1]	# 6
			target_skills_idx = deepcopy(source_skills_idx)

			if avoid_impossible:		# Skill is duplicated in a task
				if (_to in target_skills_idx) or (_from not in source_skills_idx):
					return None

			for i, sk in enumerate(target_skills_idx):
				if sk == _from:
					target_skills_idx[i] = _to

		return target_skills_idx

	@staticmethod
	def generate_random_language_guidance(*args, **kwargs):
		raise NotImplementedError()

	def get_buffer_observation(self, observation: np.ndarray):
		return observation.copy()

	def get_buffer_action(self, action: Any):
		return action.copy()

	def get_step_action(self, action: np.ndarray):
		return action.copy()

	def valid_parameter(self, *args, **kwargs):
		return True

	def get_buffer_parameter(self, parameter: np.ndarray):
		return parameter.copy()