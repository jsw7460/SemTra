import random
from typing import List, Dict, Union

import gym
import numpy as np

from comde.rl.envs.base import ComdeSkillEnv
from comde.utils.common.language_guidances import template
from meta_world.get_video import SingleTask


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

	speed_default_param = {1: 25.0, 3: 25.0, 4: 15.0, 6: 25.0}
	wind_default_param = {1: 0.0, 3: 0.0, 4: 0.0, 6: 0.0}

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		base_env = SingleTask(seed=seed, task=task)
		super(MultiStageMetaWorld, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, MultiStageMetaWorld.MW_OBS_DIM))

	def get_rtg(self):
		return self.n_target

	def step(self, action):
		obs, reward, done, info = super(MultiStageMetaWorld, self).step(action)
		if self.env.env.mode == self.n_target:
			done = True
		return obs, reward, done, info

	def get_default_parameter(self, non_functionality: str):
		if non_functionality == "speed":
			return MultiStageMetaWorld.speed_default_param
		elif non_functionality == "wind":
			return MultiStageMetaWorld.wind_default_param

		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for multistage metaworld.")


	def get_language_guidance_from_template(
		self,
		sequential_requirement: str,
		non_functionality: str,
		parameter: Union[float, Dict],
		source_skills_idx: List[int]
	) -> str:

		source_skills = [self.skill_index_mapping[sk] for sk in source_skills_idx]
		source_skills = ", and then ".join(source_skills)

		if non_functionality == "wind":
			param_applied_skill = "all skills"
			applied_param = list(parameter.values())[0]

		elif non_functionality == "speed":
			param_applied_skill = None
			applied_param = None
			for key, value in self.speed_default_param.items():
				if value != parameter[key]:
					param_applied_skill = self.skill_index_mapping[key]
					applied_param=parameter[key]
		else:
			raise NotImplementedError(f"Undefined non-functionality {non_functionality}")

		if param_applied_skill is None:
			fmt = random.choice(template[non_functionality]["default"])
		else:
			fmt = random.choice(template[non_functionality]["non_default"])

		if "replace" in sequential_requirement:
			for idx, sk in self.skill_index_mapping.items():
				sequential_requirement = sequential_requirement.replace(str(idx), str(sk))

		language_guidance = fmt.format(
			sequential_requirement=sequential_requirement,
			param_applied_skill=param_applied_skill,
			video=source_skills,
			param=applied_param
		)
		return language_guidance