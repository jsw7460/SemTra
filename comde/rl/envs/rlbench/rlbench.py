import random
from copy import deepcopy
from typing import List, Dict, Union

import gym
import numpy as np

from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.rlbench.utils import (
	OBS_DIM,
	ACT_DIM,
	RLBENCH_ALL_TASKS,
	SEQUENTIAL_REQUIREMENT,
	WEIGHT_TO_ADJECTIVE,
	SEQUENTIAL_REQUIREMENTS_VARIATIONS,
	NON_FUNCTIONALITIES_VARIATIONS,
	get_weight,
	object_in_task,
	skill_infos
)
from comde.utils.common.natural_languages.language_guidances import template
from semtra_rlbench.gym.semtra_env import SemtraEnv
from semtra_rlbench.semtra_const import (
	SEMTRA_WEIGHTS as COMDE_WEIGHTS,
	SEMTRA_PANDA_HANDLE as COMDE_PANDA_HANDLE,
	SEMTRA_PANDA_PROPERTY as COMDE_PANDA_PROPERTY
)


class DummyRLBench(gym.Env):

	def __init__(self):
		super(DummyRLBench, self).__init__()
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(387,))
		self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,))

	def step(self, action):
		pass

	def reset(self):
		pass

	def render(self, mode="human"):
		pass


class RLBench(ComdeSkillEnv):
	onehot_skills_mapping = {
		"open door": 0, "close door": 1, "close fridge": 2,
		"open drawer": 3, "close drawer": 4, "lamp off": 5,
		"push button": 6, "lamp on": 7, "press switch": 8,
		"close microwave": 9, "open box": 10, "slide block to target": 11
	}
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}
	skill_indices = list(skill_index_mapping.keys())
	non_functionalities = ["weight"]
	weight_default_param = {k: v["default"] for k, v in COMDE_WEIGHTS.items()}

	sequential_requirements_vector_mapping = None
	non_functionalities_vector_mapping = None
	has_been_called = False

	def __str__(self):
		return "rlbench"

	def __init__(
		self,
		seed: int,
		task: List[int],
		n_target: int,
		cfg: Dict = None,
		dummy: bool = False,
		register_language_embedding: bool = True
	):
		self.__t = 0
		self.idx_task = deepcopy(task)
		task = deepcopy(task)
		if not dummy:
			str_task = [RLBench.skill_index_mapping[sk] for sk in task]
			str_task = [sk.replace(" ", "_") for sk in str_task]
			skill_list = {sk: "default" for sk in str_task}  # Why list...?
			base_env = SemtraEnv(skill_list=skill_list, render_mode="rgb_array")
		else:
			base_env = DummyRLBench()

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]
		base_env.skill_list = deepcopy(task)
		super(RLBench, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,))
		self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ACT_DIM,))
		self.arm_mode_thresh = 0.0001

		if not RLBench.has_been_called:
			RLBench.has_been_called = True
			if register_language_embedding:
				mapping = self.get_sequential_requirements_mapping(SEQUENTIAL_REQUIREMENTS_VARIATIONS)
				RLBench.sequential_requirements_vector_mapping = mapping
				mapping = self.get_non_functionalities_mapping(NON_FUNCTIONALITIES_VARIATIONS)
				RLBench.non_functionalities_vector_mapping = mapping

	@staticmethod
	def get_skill_infos():
		return skill_infos

	def get_buffer_action(self, action: np.ndarray):
		action = action.copy()
		tmp0 = action[..., : 7]
		tmp1 = action[..., -3:]
		action[..., : 7] = 2 * tmp0
		action[..., -3:] = np.tanh(0.1 * np.log(tmp1))
		return action

	def get_step_action(self, action: np.ndarray):
		action = action.copy()
		tmp0 = action[..., : 7]
		tmp1 = action[..., -3:]
		action[..., : 7] = tmp0 / 2
		action[..., -3:] = np.exp(10 * np.arctanh(tmp1))
		return action

	def eval_param(self, param):
		return eval(param)

	def ingradients_to_parameter(self, prompt_extraction: str):
		pass

	def get_rtg(self):
		raise NotImplementedError()

	@staticmethod
	def get_default_parameter(non_functionality: Union[str, None] = None):
		if non_functionality == "weight":
			return deepcopy(RLBench.weight_default_param)
		elif non_functionality is None:
			default_param_dict = {nf: RLBench.get_default_parameter(nf) for nf in RLBench.non_functionalities}
			return default_param_dict

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None,
		video_parsing: bool = True
	):
		if "replace" in sequential_requirement:
			sequential_requirement = ComdeSkillEnv.replace_idx_so_skill(
				RLBench.skill_index_mapping,
				sequential_requirement
			)
		if video_parsing:
			source_skills = ComdeSkillEnv.idxs_to_str_skills(RLBench.skill_index_mapping, source_skills_idx)
			source_skills = ", then ".join(source_skills)
		else:
			source_skills = "video"

		if non_functionality == "weight":
			param_applied_skill = None
			applied_param = None
			for skill, default_weight in RLBench.weight_default_param.items():
				if default_weight != parameter[skill]:
					param_applied_skill = RLBench.skill_index_mapping[skill]
					applied_param = parameter[skill]

			if param_applied_skill is None:
				fmt = random.choice(template[non_functionality]["default"])
				language_guidance = fmt.format(video=source_skills, sequential_requirement=sequential_requirement)
			else:
				fmt = random.choice(template[non_functionality]["non_default"])
				applied_param = WEIGHT_TO_ADJECTIVE[RLBench.onehot_skills_mapping[param_applied_skill]][applied_param]
				language_guidance = fmt.format(
					video=source_skills,
					sequential_requirement=sequential_requirement,
					weight=applied_param,
					object=object_in_task(param_applied_skill)
				)
			return language_guidance
		else:
			raise NotImplementedError(f"{non_functionality} is not defined non-functionality in RL-Bench")

	@staticmethod
	def generate_random_language_guidance(video_parsing: bool = False, avoid_impossible: bool = False):
		sequential_requirement = random.choice(SEQUENTIAL_REQUIREMENT)
		non_functionality = random.choice(RLBench.non_functionalities)
		source_skills_idx = list(random.choice(RLBENCH_ALL_TASKS))

		target_skills_idx = ComdeSkillEnv.get_target_skill_from_source(
			source_skills_idx=source_skills_idx,
			sequential_requirement=sequential_requirement,
			avoid_impossible=avoid_impossible
		)

		if avoid_impossible and target_skills_idx is None:
			return None, None

		parameter = RLBench.get_default_parameter(non_functionality)

		param_applied_skill = random.choice([None] + RLBench.skill_indices)  # type: Union[None, int]

		if param_applied_skill is not None:
			weight = random.choice(["light", "heavy"])
			parameter.update({param_applied_skill: get_weight(param_applied_skill, weight)})
			param_applied_skill = object_in_task(RLBench.skill_index_mapping[param_applied_skill])

		else:
			param_applied_skill = "all"
			weight = "standard"

		language_guidance = RLBench.get_language_guidance_from_template(
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			parameter=parameter,
			source_skills_idx=source_skills_idx,
			video_parsing=video_parsing
		)
		_info = {
			"non_functionality": non_functionality,
			"param_applied_skill": param_applied_skill,
			"parameter": str(weight),
			"source_skills_idx": source_skills_idx,
			"target_skills_idx": target_skills_idx
		}

		return language_guidance, _info
