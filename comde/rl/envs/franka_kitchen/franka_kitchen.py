import random
from copy import deepcopy
from itertools import permutations
from typing import List, Dict, Union
import numpy as np

import d4rl

from comde.rl.envs.base import ComdeSkillEnv
from comde.utils.common.natural_languages.language_guidances import template
from spirl.rl.envs import kitchen
from comde.utils.common.safe_eval import safe_eval_to_float
from .utils import (
	SEQUENTIAL_REQUIREMENTS,
	SEQUENTIAL_REQUIREMENTS_VARIATIONS,
	NON_FUNCTIONALITIES_VARIATIONS,
	WIND_TO_ADJECTIVE,
	POSSIBLE_WINDS,
	ADJECTIVE_TO_WIND,
	skill_infos
)

_ = d4rl

array = np.array  # DO NOT REMOVE THIS !

class FrankaKitchen(ComdeSkillEnv):
	onehot_skills_mapping = {
		'bottom burner': 0,
		'top burner': 1,
		'light switch': 2,
		'slide cabinet': 3,
		'hinge cabinet': 4,
		'microwave': 5,
		'kettle': 6,
	}
	tasks_idxs = [0, 1, 2, 3, 4, 5, 6]
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}
	non_functionalities = ["wind"]
	wind_default_param = {k: 0.0 for k in range(7)}

	sequential_requirements_vector_mapping = None
	non_functionalities_vector_mapping = None
	has_been_called = False

	def __str__(self):
		return "kitchen"

	def __init__(
		self,
		seed: int,
		task: List,
		n_target: int,
		cfg: Dict = None,
		register_language_embedding: bool = True
	):

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		class_name = "Kitchen_"
		for skill in task:
			class_name += skill[:2]

		base_env = getattr(kitchen, class_name)
		base_env = base_env({"task_elements": tuple(task)})
		base_env.skill_list = task.copy()
		base_env._env.seed(seed)
		super(FrankaKitchen, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

		if not FrankaKitchen.has_been_called:
			FrankaKitchen.has_been_called = True
			if register_language_embedding:
				mapping = self.get_sequential_requirements_mapping(SEQUENTIAL_REQUIREMENTS_VARIATIONS)
				FrankaKitchen.sequential_requirements_vector_mapping = mapping
				mapping = self.get_non_functionalities_mapping(NON_FUNCTIONALITIES_VARIATIONS)
				FrankaKitchen.non_functionalities_vector_mapping = mapping

	@staticmethod
	def get_skill_infos():
		return skill_infos

	def set_str_parameter(self, parameter: str):
		print("Set parameter !" * 999, parameter)
		if (parameter not in ["breeze", "default", "gust", "flurry"]) or (type(parameter) != str):
			raise NotImplementedError(f"{parameter} is not supported for Franka kitchen environment.")

		self.str_parameter = parameter

	def eval_param(self, param):
		return safe_eval_to_float(param)

	def get_rtg(self):
		return self.n_target

	def step(self, action: np.ndarray):
		if self.str_parameter in ["default", "breeze"]:
			pass
		elif self.str_parameter == "gust":
			action[0] -= 0.1
		elif self.str_parameter == "flurry":
			action[0] -= 0.3
		else:
			raise NotImplementedError(f"{self.str_parameter} is not supported in Franka kitchen environment.")

		obs, rew, done, info = super(FrankaKitchen, self).step(action)
		return obs, rew, done, info

	@staticmethod
	def get_parameter_from_adjective(adjective: str):
		if adjective.lower() == "default":
			adjective = "breeze"

		if adjective.lower() in ["breeze", "gust", "flurry"]:
			return {k: eval(ADJECTIVE_TO_WIND[adjective]) for k in range(7)}

		else:
			raise NotImplementedError(f"Adjective {adjective} is not supported in Franka kitchen environment.")

	@staticmethod
	def get_default_parameter(non_functionality: Union[str, None] = None):
		if non_functionality == "wind":
			return deepcopy(FrankaKitchen.wind_default_param)
		elif non_functionality is None:
			default_param_dict = {
				nf: FrankaKitchen.get_default_parameter(nf) for nf in FrankaKitchen.non_functionalities
			}
			return default_param_dict
		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for franka kitchen.")

	def valid_parameter(self, param_dict: Dict):
		for v in param_dict.values():
			if v not in POSSIBLE_WINDS:
				return False
		return True

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None,
		video_parsing: bool = True
	):
		if parameter is None:
			parameter = FrankaKitchen.get_default_parameter(non_functionality)

		if video_parsing:
			source_skills = ComdeSkillEnv.idxs_to_str_skills(FrankaKitchen.skill_index_mapping, source_skills_idx)
			source_skills = ", and then ".join(source_skills)
		else:
			source_skills = "video"

		if non_functionality == "wind":
			fmt = random.choice(template["wind"]["non_default"])
			parameters = list(parameter.values())
			wind = parameters[0]
			param = WIND_TO_ADJECTIVE.get(str(wind), None)
			if param is None:
				return None

		else:
			raise NotImplementedError()

		if "replace" in sequential_requirement:
			sequential_requirement = ComdeSkillEnv.replace_idx_so_skill(
				FrankaKitchen.skill_index_mapping,
				sequential_requirement
			)

		language_guidance = fmt.format(
			sequential_requirement=sequential_requirement,
			video=source_skills,
			param=param
		)

		return language_guidance

	@staticmethod
	def generate_random_language_guidance(video_parsing: bool = False, avoid_impossible: bool = False):
		tasks = deepcopy(FrankaKitchen.tasks_idxs)
		sequential_requirement = random.choice(SEQUENTIAL_REQUIREMENTS)
		non_functionality = "wind"
		perm = list(permutations(tasks, 4))
		perm = [list(p) for p in perm]
		source_skills_idx = list(random.choice(perm))

		target_skills_idx = ComdeSkillEnv.get_target_skill_from_source(
			source_skills_idx=source_skills_idx,
			sequential_requirement=sequential_requirement,
			avoid_impossible=avoid_impossible
		)

		if avoid_impossible and target_skills_idx is None:
			return None, None

		param_applied_skill = "all"
		wind = random.choice(POSSIBLE_WINDS)
		parameter = {k: wind for k in tasks}
		param_for_apply = WIND_TO_ADJECTIVE[str(wind)]

		language_guidance = FrankaKitchen.get_language_guidance_from_template(
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			source_skills_idx=list(source_skills_idx),
			parameter=parameter,
			video_parsing=video_parsing
		)
		_info = {
			"non_functionality": non_functionality,
			"param_applied_skill": param_applied_skill,
			"parameter": str(param_for_apply),
			"int_parameter": param_for_apply,
			"source_skills_idx": source_skills_idx,
			"target_skills_idx": target_skills_idx
		}
		return language_guidance, _info

	def ingradients_to_parameter(self, ingradients: Dict[str, str], scale: bool = True):

		non_functionality = ingradients["non_functionality"]
		param_applied_skill = ingradients["skill"]
		param = ingradients["param"]

		is_nf_wrong = non_functionality not in self.non_functionalities
		is_skill_wrong = param_applied_skill not in list(self.onehot_skills_mapping.keys()) + ["all"]

		if is_nf_wrong:
			return self.wind_default_param

		else:
			parameter = self.get_default_parameter(non_functionality)

			if is_skill_wrong or (param == "standard"):
				return parameter

			else:
				param = ADJECTIVE_TO_WIND[param]
				if param_applied_skill == "all":
					for k in parameter.keys():
						parameter[k] = param

				elif param_applied_skill != "all":
					parameter.update({self.onehot_skills_mapping[param_applied_skill]: param})

		return parameter