from typing import List, Dict

from comde.rl.envs.base import ComdeSkillEnv
from spirl.rl.envs import kitchen


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

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		class_name = "Kitchen_"
		for skill in task:
			class_name += skill[:2]

		base_env = getattr(kitchen, class_name)
		base_env = base_env({})
		base_env.skill_list = task.copy()
		super(FrankaKitchen, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

	def step(self, action):
		obs, rew, done, info = super(FrankaKitchen, self).step(action)
		return obs, rew, done, info