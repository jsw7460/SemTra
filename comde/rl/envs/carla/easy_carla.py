import pickle
import sys

sys.path.append("/home/jsw7460/comde/easy_carla/")

from easy_carla.carla_env import SkillAttatchedCarlaEnvironment
from easy_carla.utils.config import ExperimentConfigs, LidarConfigs
from typing import List, Dict, Union
from carla import VehicleControl

import gym
import numpy as np

from comde.rl.envs.base import ComdeSkillEnv


class EasyCarla(ComdeSkillEnv):
	"""
		Actual carla environment
	"""
	# Image embedding dim = 1000 (Resnet 50)
	# Sensor dim = 104
	OBSERVATION_DIM = 1104	# Image embedding dim + Sensor dim
	ACTION_DIM = 3
	onehot_skills_mapping = {
		'stop': 0,
		'straight': 1,
		'left': 2,
		'right': 3,
	}
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):
		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		self._img_embedding = None	# Required if we have to embed raw image
		base_env = self.get_base_env(cfg)
		base_env.skill_list = task.copy()

		super(EasyCarla, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

		self.dotmap_observation_space = self.observation_space
		self.dotmap_action_space = self.action_space

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBSERVATION_DIM, ))
		self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(self.ACTION_DIM, ))

		self.params_dict = None
		self.load_param_dict()

	def get_rtg(self):
		raise NotImplementedError("Implement return-to-go for Carla environment.")

	def get_base_env(self, cfg: Dict) -> gym.Env:
		from comde.utils.common.pretrained_forward import resnet50_forward
		self._img_embedding = resnet50_forward
		carla_cfg = cfg["carla_cfg"]
		exp_configs = carla_cfg.pop("config")
		exp_configs["vehicle_type"] = "audi.a2"
		exp_configs["lidar"] = LidarConfigs(**exp_configs["lidar"])
		exp_configs["max_steps"] = 3000
		config = ExperimentConfigs(**exp_configs)
		return SkillAttatchedCarlaEnvironment(config=config, **carla_cfg)

	def load_param_dict(self):
		with open(self.cfg["params_dict_path"], "rb") as f:
			self.params_dict = pickle.load(f)

	def postprocess_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray]):
		img = obs["image"]
		img = img[np.newaxis, ...]
		emb = self._img_embedding(img)
		emb = np.squeeze(emb, axis=0)
		sensor = obs["sensor"]
		processed_obs = np.concatenate((emb, sensor), axis=-1)
		return processed_obs

	@staticmethod
	def postprocess_action(action: np.ndarray):
		assert action.shape[-1] == 3
		control = VehicleControl(
			throttle=float(action[0]),
			steer=float(action[1]),
			brake=float(action[2]),
			hand_brake=False,
			reverse=False,
			manual_gear_shift=False,
			gear=int(0)
		)
		return control

	def get_buffer_action(self, action: np.ndarray) -> np.ndarray:
		optimal_action, _ = self.compute_action()
		optimal_action = np.array([optimal_action.throttle, optimal_action.steer, optimal_action.brake])
		action[0] = optimal_action[0]
		# action[1] = optimal_action[1]
		action[2] = optimal_action[2]
		return action
		# if self.count < 100000:
		# 	action, _ = self.compute_action()
		# 	return np.array([action.throttle, action.steer, action.brake])
		# else:
		# 	action[0] = 0.75
		# 	return action

	def step(self, action: np.ndarray):
		if self.count < 10:
			print("Optimal action", action)
		else:
			print("Predict action", action)
		vehicle_control = self.postprocess_action(action)
		obs, rew, done, info = super(EasyCarla, self).step(vehicle_control)
		obs = self.postprocess_observation(obs)
		return obs, rew, done, info

	def reset(self, **kwargs):
		obs = super(EasyCarla, self).reset(**kwargs)
		return self.postprocess_observation(obs)

	def get_default_parameter(self, non_functionality: str):
		# return self.params_dict["carlamotors.carlacola"]
		return self.params_dict["audi.a2"]


class DummyEasyCarla(EasyCarla):
	"""
		Dummy carla environment
	"""

	def step(self, action):
		raise NotImplementedError("This is not for environment interaction")

	def reset(self, **kwargs):
		raise NotImplementedError("This is not for environment interaction")

	def get_base_env(self, cfg: Dict) -> gym.Env:
		return _ObservationalDummyCarla()

	def load_param_dict(self):
		pass


class _ObservationalDummyCarla(gym.Env):
	def __init__(self):
		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(1, EasyCarla.OBSERVATION_DIM)
		)
		self.action_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(1, EasyCarla.ACTION_DIM)
		)

	def step(self, action):
		raise NotImplementedError("This environment is not for interaction")

	def reset(self):
		raise NotImplementedError("This environment is not for interaction")

	def render(self, mode="human"):
		raise NotImplementedError("This environment is not for interaction")
