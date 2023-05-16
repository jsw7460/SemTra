from typing import List, Dict

from comde.rl.envs.base import ComdeSkillEnv
from spirl.rl.envs import kitchen
import d4rl

_ = d4rl


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

	def get_default_parameter(self, non_functionality: str):
		if non_functionality == "wind":
			return FrankaKitchen.wind_default_param
		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for franka kitchen.")
