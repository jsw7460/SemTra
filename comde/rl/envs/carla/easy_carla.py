import pickle
import sys
from copy import deepcopy

sys.path.append("/home/jsw7460/comde/easy_carla/")

from easy_carla.carla_env import SkillAttatchedCarlaEnvironment
from easy_carla.utils.config import ExperimentConfigs, LidarConfigs
from typing import List, Dict, Union
from carla import VehicleControl

import gym
import numpy as np

from comde.rl.envs.base import ComdeSkillEnv
from .utils import (
	skill_infos,
	SEQUENTIAL_REQUIREMENTS_VARIATIONS,
	NON_FUNCTIONALITIES_VARIATIONS
)

array = np.array  # DO NOT REMOVE THIS !
EPS = 1e-12


class EasyCarla(ComdeSkillEnv):
	"""
		Actual carla environment
	"""
	# Image embedding dim = 1000 (Resnet 50)
	# Sensor dim = 104
	OBSERVATION_DIM = 1104  # Image embedding dim + Sensor dim
	ACTION_DIM = 3
	default_vehicle = "audi.a2"
	onehot_skills_mapping = {
		'stop': 0,
		'straight': 1,
		'left': 2,
		'right': 3,
	}
	non_functionalities = ["vehicle"]
	skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}
	vehicle_default_param = None

	sequential_requirements_vector_mapping = None
	non_functionalities_vector_mapping = None
	has_been_called = False

	def __init__(self, seed: int, task: List, n_target: int, cfg: Dict = None):

		self.port = cfg.get("port", None)
		self.observation_keys = cfg["observation_keys"]

		with open(cfg["normalization_path"], "rb") as f:
			normalization_dict = pickle.load(f)

		self.sensor_max = normalization_dict["sensor_max"]
		self.sensor_min = normalization_dict["sensor_min"]
		self.img_emb_max = normalization_dict["img_emb_max"]
		self.img_emb_min = normalization_dict["img_emb_min"]
		self.action_max = normalization_dict["action_max"]
		self.action_min = normalization_dict["action_min"]
		self.param_max = normalization_dict["param_max"]
		self.param_min = normalization_dict["param_min"]

		self.normalization_dict = {
			"img_embedding": {"max": self.img_emb_max, "min": self.img_emb_min},
			"sensors": {"max": self.sensor_max, "min": self.sensor_min}
		}

		assert list(self.normalization_dict.keys()) == self.observation_keys

		obs_max = []
		obs_min = []
		for key in self.observation_keys:
			obs_max.append(self.normalization_dict[key]["max"])
			obs_min.append(self.normalization_dict[key]["min"])

		self.obs_max = np.hstack(obs_max)
		self.obs_min = np.hstack(obs_min)

		if type(task[0]) == int:
			for i in range(len(task)):
				task[i] = self.skill_index_mapping[task[i]]

		self._img_embedding = None  # Required if we have to embed raw image
		base_env = self.get_base_env(cfg)
		base_env.skill_list = task.copy()

		super(EasyCarla, self).__init__(env=base_env, seed=seed, task=task, n_target=n_target, cfg=cfg)

		self.dotmap_observation_space = self.observation_space
		self.dotmap_action_space = self.action_space

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBSERVATION_DIM,))
		self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(self.ACTION_DIM,))

		self.params_dict = None
		self.load_param_dict()

		if not EasyCarla.has_been_called:
			EasyCarla.has_been_called = True
			mapping = self.get_sequential_requirements_mapping(SEQUENTIAL_REQUIREMENTS_VARIATIONS)
			EasyCarla.sequential_requirements_vector_mapping = mapping

			mapping = self.get_non_functionalities_mapping(NON_FUNCTIONALITIES_VARIATIONS)
			EasyCarla.non_functionalities_vector_mapping = mapping

	def get_rtg(self):
		raise NotImplementedError("Implement return-to-go for Carla environment.")

	def get_base_env(self, cfg: Dict) -> gym.Env:
		from comde.utils.common.pretrained_forwards.th_resnet_50 import resnet50_forward
		self._img_embedding = resnet50_forward
		carla_cfg = cfg["carla_cfg"]
		exp_configs = carla_cfg.pop("config")
		exp_configs["vehicle_type"] = "audi.a2"
		exp_configs["lidar"] = LidarConfigs(**exp_configs["lidar"])
		exp_configs["max_steps"] = 3000
		if self.port is not None:
			carla_cfg["carla_port"] = self.port	# Override
			exp_configs["carla_port"] = self.port	# Override
			config = ExperimentConfigs(**exp_configs)
		else:
			config = ExperimentConfigs(**exp_configs)
		return SkillAttatchedCarlaEnvironment(config=config, **carla_cfg)

	def load_param_dict(self):
		with open(self.cfg["params_dict_path"], "rb") as f:
			self.params_dict = pickle.load(f)

		EasyCarla.vehicle_default_param = self.params_dict[EasyCarla.default_vehicle]

	def postprocess_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray]):
		img = obs["image"]
		img = img[np.newaxis, ...]
		emb = self._img_embedding(img)
		emb = np.squeeze(emb, axis=0)
		sensor = obs["sensor"]
		processed_obs = np.concatenate((emb, sensor), axis=-1)

		return self.get_buffer_observation(processed_obs)

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

	@staticmethod
	def get_language_guidance_from_template(
		sequential_requirement: str,
		non_functionality: str,
		source_skills_idx: List[int],
		parameter: Union[float, Dict] = None,
		video_parsing: bool = True
	):
		print("Sequential requirement", sequential_requirement)
		print("Non functionality", non_functionality)
		print("Parameter", parameter)
		print("Video parsing", video_parsing)
		exit()

	def get_buffer_observation(self, observation: np.ndarray):
		# assert np.all(observation >= self.obs_min) and np.all(observation <= self.obs_max), \
		# 		f"Observation: {observation}, Min: {self.obs_min}, Max: {self.obs_max}"
		return (observation - self.obs_min) / (self.obs_max - self.obs_min + EPS)

	def get_buffer_action(self, action: np.ndarray):
		return action
		# assert np.all(action >= self.action_min) and np.all(action <= self.action_max)
		# action = (action - self.action_min) / (self.action_max - self.action_min)
		# return action

	def get_step_action(self, action: np.ndarray):
		return action * (self.action_max - self.action_min) + self.action_min

	def get_buffer_parameter(self, parameter: np.ndarray):
		# print("What is the parameter type?", parameter.dtype)
		# print("Before 5th", parameter[..., 5])
		parameter = (parameter - self.param_min) / (self.param_max - self.param_min + 1E-9)
		# print("Dominator", (parameter - self.param_min)[..., 5])
		# print("After 5th", parameter[..., 5])
		parameter[parameter > 0.5] = 1.0
		parameter[parameter < 0.5] = 0.0
		# print("Output parameter?", parameter)
		return parameter

	def step(self, action: np.ndarray):
		action = self.get_step_action(action)
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

	@staticmethod
	def get_skill_infos():
		return deepcopy(skill_infos)

	def eval_param(self, param):
		return eval(param)

	@staticmethod
	def get_default_parameter(non_functionality: Union[str, None] = None):
		if non_functionality == "vehicle":
			return deepcopy(EasyCarla.vehicle_default_param)
		elif non_functionality is None:
			default_param_dict = {
				nf: EasyCarla.get_default_parameter(nf) for nf in EasyCarla.non_functionalities
			}
			return default_param_dict
		else:
			raise NotImplementedError(f"{non_functionality} is undefined non functionality for CARLA.")


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
		super(DummyEasyCarla, self).load_param_dict()


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
