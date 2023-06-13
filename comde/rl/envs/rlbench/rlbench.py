import random
from copy import deepcopy
from typing import List, Dict, Union

from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.rlbench.utils import (
	get_task_class,
	get_weight,
	object_in_task,
	RLBENCH_ALL_TASKS,
	SEQUENTIAL_REQUIREMENT,
	WEIGHT_TO_ADJECTIVE
)
from comde_rlbench.RLBench.comde.comde_env import ComdeEnv as ComdeRLBench
from comde_rlbench.RLBench.rlbench.comde_const import COMDE_WEIGHTS
from comde.utils.common.language_guidances import template


class RLBench(ComdeSkillEnv):
	onehot_skills_mapping = {
		"open door": 0, "close door": 1, "close fridge": 2,
		"open drawer": 3, "close drawer": 4, "lamp off": 5,
		"push button": 6, "lamp on": 7, "press switch": 8,
		"close microwave": 9, "open box": 10, "slide block to target": 11,
	}
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}
	skill_indices = list(skill_index_mapping.keys())
	non_functionalities = ["weight"]
	# 0 is the default weight index
	weight_default_param = {k: v[0] for k, v in COMDE_WEIGHTS.items()}

	def __init__(self, seed: int, task: List[int], n_target: int, cfg: Dict = None):
		task_class = get_task_class(task)
		base_env = ComdeRLBench(task_class=task_class, place_seq=task)
		super(RLBench, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

	def ingradients_to_parameter(self, prompt_extraction: str):
		pass

	def get_rtg(self):
		raise NotImplementedError()

	@staticmethod
	def get_default_parameter(non_functionality: str):
		if non_functionality == "weight":
			return deepcopy(RLBench.weight_default_param)

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		parameter: Union[float, Dict],
		source_skills_idx: List[int],
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
	def generate_random_language_guidance():
		sequential_requirement = random.choice(SEQUENTIAL_REQUIREMENT)
		non_functionality = random.choice(RLBench.non_functionalities)
		source_skills_idx = random.choice(RLBENCH_ALL_TASKS)

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
			video_parsing=False
		)


		_info = {
			"non_functionality": non_functionality,
			"param_applied_skill": param_applied_skill,
			"parameter": str(weight)
		}

		return language_guidance, _info