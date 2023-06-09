import math
import random
from copy import deepcopy
from itertools import permutations
from typing import List, Dict, Union, Tuple

import gym
import numpy as np

from comde.rl.envs.base import ComdeSkillEnv
from comde.utils.common.language_guidances import template
from meta_world.get_video import SingleTask

SEQUENTIAL_REQUIREMENTS = [
	"sequential",
	"reverse"
]
for (a, b) in list(permutations(range(8), 2)):
	SEQUENTIAL_REQUIREMENTS.append(f"replace {a} with {b}")

POSSIBLE_SPEEDS = {
	1: {"default": 25.0, "non_default": {"normal": 10.0, "slow": 1.5}},
	3: {"default": 25.0, "non_default": {"normal": 13.0, "slow": 6.0}},
	4: {"default": 15.0, "non_default": {"normal": 5.0, "slow": 1.5}},
	6: {"default": 25.0, "non_default": {"normal": 8.0, "slow": 3.0}}
}
POSSIBLE_WINDS = [
	-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3
]


class MultiStageMetaWorld(ComdeSkillEnv):
	MW_OBS_DIM = 140
	onehot_skills_mapping = {
		"box": 0,
		"puck": 1,
		"handle": 2,
		"drawer": 3,
		"button": 4,
		"lever": 5,
		"door": 6,
		"stick": 7
	}
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}

	TASKS_IDXS = {
		"easy": [1, 3, 4, 6]
	}

	speed_default_param = {1: 25.0, 3: 25.0, 4: 15.0, 6: 25.0}
	wind_default_param = {1: 0.0, 3: 0.0, 4: 0.0, 6: 0.0}

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		base_env = SingleTask(seed=seed, task=task)
		super(MultiStageMetaWorld, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, MultiStageMetaWorld.MW_OBS_DIM))
		if cfg is None:
			self.difficulty = "easy"
		else:
			self.difficulty = cfg.get("difficulty", "easy")

	def get_rtg(self):
		return self.n_target

	def step(self, action):
		obs, reward, done, info = super(MultiStageMetaWorld, self).step(action)
		if self.env.env.mode == self.n_target:
			done = True
		return obs, reward, done, info

	@staticmethod
	def get_default_parameter(non_functionality: str):
		if non_functionality == "speed":
			return MultiStageMetaWorld.speed_default_param
		elif non_functionality == "wind":
			return MultiStageMetaWorld.wind_default_param
		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for multistage metaworld.")

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None
	) -> str:

		if parameter is None:
			parameter = {1: 1.5, 3: 25.0, 4: 15.0, 6: 25.0}

		source_skills = ComdeSkillEnv.idxs_to_str_skills(MultiStageMetaWorld.skill_index_mapping, source_skills_idx)
		source_skills = ", and then ".join(source_skills)

		if non_functionality == "wind":
			param_applied_skill = "all skills"
			applied_param = list(parameter.values())[0]

		elif non_functionality == "speed":
			param_applied_skill = None
			applied_param = None
			for key, value in MultiStageMetaWorld.speed_default_param.items():
				if value != parameter[key]:
					param_applied_skill = MultiStageMetaWorld.skill_index_mapping[key]
					applied_param = parameter[key]
		else:
			raise NotImplementedError(f"Undefined non-functionality {non_functionality}")

		if param_applied_skill is None:
			fmt = random.choice(template[non_functionality]["default"])
		else:
			fmt = random.choice(template[non_functionality]["non_default"])

		if "replace" in sequential_requirement:
			sequential_requirement = ComdeSkillEnv.replace_idx_so_skill(
				MultiStageMetaWorld.skill_index_mapping,
				sequential_requirement
			)

		language_guidance = None
		if non_functionality == "wind":
			if applied_param > 0:
				_from = "west"
				_to = "east"
			else:
				_from = "east"
				_to = "west"

			param = math.fabs(applied_param)

			language_guidance = fmt.format(
				sequential_requirement=sequential_requirement,
				video=source_skills,
				_from=_from,
				_to=_to,
				param=int(param * 10)
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
	def generate_random_language_guidance(difficulty: str = "easy") -> Tuple[str, Dict]:
		tasks = deepcopy(MultiStageMetaWorld.TASKS_IDXS[difficulty])
		sequential_requirement = random.choice(SEQUENTIAL_REQUIREMENTS)
		non_functionality = random.choices(["speed", "wind"], weights=[100, 10])[0]
		source_skills_idx = list(random.choice(list(permutations(tasks, 3))))

		if non_functionality == "speed":
			param_applied_skill = random.choice(["all skills"] + tasks)
			parameter = deepcopy(MultiStageMetaWorld.speed_default_param)

			if param_applied_skill != "all skills":
				param_for_apply = random.choice(list(POSSIBLE_SPEEDS[param_applied_skill]["non_default"].values()))
				parameter.update(
					{param_applied_skill: param_for_apply}
				)
				param_applied_skill = MultiStageMetaWorld.skill_index_mapping[param_applied_skill]
			else:
				param_for_apply = "standard"

		elif non_functionality == "wind":
			param_applied_skill = "all skills"
			wind = random.choice(POSSIBLE_WINDS)
			parameter = {k: wind for k in tasks}
			param_for_apply = int(wind * 10)

		else:
			raise NotImplementedError()

		language_guidance = MultiStageMetaWorld.get_language_guidance_from_template(
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			source_skills_idx=source_skills_idx,
			parameter=parameter
		)

		_info = {
			"non_functionality": non_functionality,
			"param_applied_skill": param_applied_skill,
			"parameter": str(param_for_apply)
		}

		return language_guidance, _info
