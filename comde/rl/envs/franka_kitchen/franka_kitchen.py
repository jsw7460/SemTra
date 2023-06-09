import random
from copy import deepcopy
from itertools import permutations
from typing import List, Dict, Union

import math

import d4rl

from comde.rl.envs.base import ComdeSkillEnv
from comde.utils.common.language_guidances import template
from spirl.rl.envs import kitchen

_ = d4rl

SEQUENTIAL_REQUIREMENTS = [
	"sequential",
	"reverse"
]
for (a, b) in list(permutations(range(7), 2)):
	SEQUENTIAL_REQUIREMENTS.append(f"replace {a} with {b}")


POSSIBLE_WINDS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]



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
	TASKS_IDXS = [0, 1, 2, 3, 4, 5, 6]
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}
	wind_default_param = {k: 0.0 for k in range(7)}

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

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

	def get_rtg(self):
		return self.n_target

	def step(self, action):
		obs, rew, done, info = super(FrankaKitchen, self).step(action)
		return obs, rew, done, info

	@staticmethod
	def get_default_parameter(non_functionality: str):
		if non_functionality == "wind":
			return FrankaKitchen.wind_default_param
		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for franka kitchen.")

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None,
	):
		if parameter is None:
			parameter = FrankaKitchen.get_default_parameter(non_functionality)
		source_skills = ComdeSkillEnv.idxs_to_str_skills(FrankaKitchen.skill_index_mapping, source_skills_idx)
		source_skills = ", and then ".join(source_skills)

		if non_functionality == "wind":
			fmt = random.choice(template["wind"]["non_default"])
			parameters = list(parameter.values())
			wind = parameters[0]

			# Wind is + : west -> east // Wind is - : east -> west
			if wind > 0:
				_from = "west"
				_to = "east"
			else:
				_from = "east"
				_to = "west"
			param = math.fabs(wind)

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
			_from=_from,
			_to=_to,
			param=int(param * 10)
		)

		return language_guidance

	@staticmethod
	def generate_random_language_guidance():
		tasks = deepcopy(FrankaKitchen.TASKS_IDXS)
		sequential_requirement = random.choice(SEQUENTIAL_REQUIREMENTS)
		non_functionality = "wind"
		source_skills_idx = random.choice(list(permutations(tasks, 4)))

		param_applied_skill = "all skills"
		wind = random.choice(POSSIBLE_WINDS)
		parameter = {k: wind for k in tasks}
		param_for_apply = int(wind * 10)

		language_guidance = FrankaKitchen.get_language_guidance_from_template(
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			source_skills_idx=list(source_skills_idx),
			parameter=parameter
		)
		_info = {
			"non_functionality": non_functionality,
			"param_applied_skill": param_applied_skill,
			"parameter": str(param_for_apply)
		}
		# print(language_guidance, _info)
		# print("\n\n\n")
		return language_guidance, _info