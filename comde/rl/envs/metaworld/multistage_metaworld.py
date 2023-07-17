import random
from copy import deepcopy
from itertools import permutations
from typing import List, Dict, Union, Tuple
import math

import gym
import numpy as np

from comde.rl.envs.base import ComdeSkillEnv
from comde.utils.common.natural_languages.language_guidances import template
from meta_world.get_video import SingleTask
from .utils import (
	SPEED_TO_ADJECTIVE,
	WIND_TO_ADJECTIVE,
	ADJECTIVE_TO_SPEED,
	SEQUENTIAL_REQUIREMENTS,
	POSSIBLE_WINDS,
	POSSIBLE_SPEEDS,
	IDX_TO_PARAMETERS,
	SCALE,
	SEQUENTIAL_REQUIREMENTS_VARIATIONS,
	NON_FUNCTIONALITIES_VARIATIONS,
	skill_infos
)


class MultiStageMetaWorld(ComdeSkillEnv):
	mw_obs_dim = 140
	# tasks_idxs = {"easy": [1, 3, 4, 6]}		# 원래 이거임 !!!!!!!!
	tasks_idxs = {"easy": [1, 2, 3, 4, 5, 6]}
	onehot_skills_mapping = {
		"box": 0, "puck": 1, "handle": 2, "drawer": 3, "button": 4, "lever": 5, "door": 6, "stick": 7
	}
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}

	non_functionalities = ["wind", "speed"]
	speed_default_param = {1: 25.0, 2: 25.0, 3: 25.0, 4: 15.0, 5: 15.0, 6: 25.0}
	wind_default_param = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
	param_dim = 1

	sequential_requirements_vector_mapping = None
	non_functionalities_vector_mapping = None
	has_been_called = False

	idx_to_parameter = {0: 0.1, 1: 0.2, 2: 0.3}

	def __str__(self):
		return "metaworld"

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

		base_env = SingleTask(seed=seed, task=task)
		super(MultiStageMetaWorld, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, MultiStageMetaWorld.mw_obs_dim))
		if cfg is None:
			self.difficulty = "easy"
		else:
			self.difficulty = cfg.get("difficulty", "easy")
		self.timestep_per_skills = {k: 0 for k in task}

		if not MultiStageMetaWorld.has_been_called:
			MultiStageMetaWorld.has_been_called = True
			if register_language_embedding:
				mapping = self.get_sequential_requirements_mapping(SEQUENTIAL_REQUIREMENTS_VARIATIONS)
				MultiStageMetaWorld.sequential_requirements_vector_mapping = mapping
				mapping = self.get_non_functionalities_mapping(NON_FUNCTIONALITIES_VARIATIONS)
				MultiStageMetaWorld.non_functionalities_vector_mapping = mapping

	def get_idx_from_parameter(self, skill_idx: int, parameter: float):
		param_dict = IDX_TO_PARAMETERS[int(skill_idx)]
		for k, v in param_dict.items():
			if math.fabs(parameter - v) < 1e-9:
				return k
		raise NotImplementedError()

	def get_parameter_from_idx(self):
		raise NotImplementedError()

	def get_idx_to_parameter_dict(self):
		return deepcopy(IDX_TO_PARAMETERS)

	def reset(self, **kwargs):
		self.timestep_per_skills = {k: 0 for k in self.task}
		return super(MultiStageMetaWorld, self).reset(**kwargs)

	def get_parameter_from_adjective(self, adjective: str):

		if adjective == "default":
			adjective = "fast"		# In metaworld, just do this. (I made a mistake for naming when I made a dataset)
		if adjective not in ["normal", "slow", "fast"]:
			raise NotImplementedError(f"Parameter {adjective} is not supported in metaworld !")

		# Make first subtask (skill) to be applied by the parameter
		# This is just for convenience of the experiments.
		param_dict = MultiStageMetaWorld.get_default_parameter("speed")
		first_skill = MultiStageMetaWorld.onehot_skills_mapping[self.task[0]]
		param_dict[first_skill] = ADJECTIVE_TO_SPEED[first_skill][adjective]

		return param_dict

	@staticmethod
	def get_skill_infos():
		return deepcopy(skill_infos)

	def eval_param(self, param):
		return eval(param)

	def get_rtg(self):
		return self.n_target

	def step(self, action):
		obs, reward, done, info = super(MultiStageMetaWorld, self).step(action)
		self.timestep_per_skills[self.task[self.env.mode]] += 1

		if self.env.env.mode == self.n_target:
			done = True
		return obs, reward, done, info

	@staticmethod
	def get_default_parameter(non_functionality: Union[str, None] = None):
		if non_functionality == "speed":
			return deepcopy(MultiStageMetaWorld.speed_default_param)
		elif non_functionality == "wind":
			return deepcopy(MultiStageMetaWorld.wind_default_param)
		elif non_functionality is None:
			default_param_dict = {
				nf: MultiStageMetaWorld.get_default_parameter(nf) for nf in MultiStageMetaWorld.non_functionalities
			}
			return default_param_dict
		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for multistage metaworld.")

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None,
		video_parsing: bool = True
	) -> str:
		if parameter is None:
			parameter = MultiStageMetaWorld.get_default_parameter(non_functionality)

		if video_parsing:
			source_skills = ComdeSkillEnv.idxs_to_str_skills(MultiStageMetaWorld.skill_index_mapping, source_skills_idx)
			source_skills = ", then ".join(source_skills)
		else:
			source_skills = "video"

		if non_functionality == "wind":
			param_applied_skill = "all"
			applied_param = list(parameter.values())[0]
			fmt = random.choice(template[non_functionality]["non_default"])
		elif non_functionality == "speed":
			param_applied_skill = None
			applied_param = None
			for key, value in MultiStageMetaWorld.speed_default_param.items():
				if value != parameter[key]:
					param_applied_skill = MultiStageMetaWorld.skill_index_mapping[key]
					applied_param = parameter[key]

			if param_applied_skill is None:
				fmt = random.choice(template[non_functionality]["default"])
				applied_param = 0
			else:
				fmt = random.choice(template[non_functionality]["non_default"])
				applied_param \
					= SPEED_TO_ADJECTIVE[
					MultiStageMetaWorld.onehot_skills_mapping[param_applied_skill]
				][str(applied_param)]
		else:
			raise NotImplementedError(f"Undefined non-functionality {non_functionality}")

		if "replace" in sequential_requirement:
			sequential_requirement = ComdeSkillEnv.replace_idx_so_skill(
				MultiStageMetaWorld.skill_index_mapping,
				sequential_requirement
			)

		language_guidance = None
		if non_functionality == "wind":
			param = WIND_TO_ADJECTIVE[str(applied_param)]

			language_guidance = fmt.format(
				sequential_requirement=sequential_requirement,
				video=source_skills,
				param=param
			)

		elif non_functionality == "speed":
			language_guidance = fmt.format(
				sequential_requirement=sequential_requirement,
				param_applied_skill=param_applied_skill,
				video=source_skills,
				param=applied_param
			)

		return language_guidance

	@staticmethod
	def generate_random_language_guidance(
		difficulty: str = "easy",
		video_parsing: bool = False,
		avoid_impossible: bool = False
	) -> Union[Tuple[str, Dict], Tuple[None, None]]:
		tasks = deepcopy(MultiStageMetaWorld.tasks_idxs[difficulty])
		sequential_requirement = random.choice(SEQUENTIAL_REQUIREMENTS)
		non_functionality = "speed"
		source_skills_idx = list(random.choice(list(permutations(tasks, 3))))

		target_skills_idx = ComdeSkillEnv.get_target_skill_from_source(
			source_skills_idx=source_skills_idx,
			sequential_requirement=sequential_requirement,
			avoid_impossible=avoid_impossible
		)

		if avoid_impossible and target_skills_idx is None:
			return None, None

		if non_functionality == "speed":
			param_applied_skill = random.choice(["all"] + tasks)
			parameter = deepcopy(MultiStageMetaWorld.speed_default_param)

			if param_applied_skill != "all":
				param_for_apply = random.choice(list(POSSIBLE_SPEEDS[param_applied_skill]["non_default"].values()))
				parameter.update(
					{param_applied_skill: param_for_apply}
				)
				param_for_apply = param_for_apply  # Scaling
				param_for_apply = SPEED_TO_ADJECTIVE[param_applied_skill][str(param_for_apply)]
				param_applied_skill = MultiStageMetaWorld.skill_index_mapping[param_applied_skill]
			else:
				param_for_apply = "standard"

		elif non_functionality == "wind":
			param_applied_skill = "all"
			wind = random.choice(POSSIBLE_WINDS)
			parameter = {k: wind for k in tasks}
			param_for_apply = WIND_TO_ADJECTIVE[str(wind)]

		else:
			raise NotImplementedError()

		language_guidance = MultiStageMetaWorld.get_language_guidance_from_template(
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			source_skills_idx=source_skills_idx,
			parameter=parameter,
			video_parsing=video_parsing
		)

		_info = {
			"non_functionality": non_functionality,
			"param_applied_skill": param_applied_skill,
			"parameter": str(param_for_apply),
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
			return random.choice([self.speed_default_param, self.wind_default_param])

		else:
			parameter = self.get_default_parameter(non_functionality)
			if is_skill_wrong or (param == "standard"):
				return parameter

			else:
				param = ADJECTIVE_TO_SPEED[self.onehot_skills_mapping[param_applied_skill]][param]
				if scale:
					param /= SCALE
				if param_applied_skill != "all":
					parameter.update({self.onehot_skills_mapping[param_applied_skill]: param})

		return parameter
