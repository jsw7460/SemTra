from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Tuple

import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize

from comde.rl.buffers.type_aliases import ReplayBufferSamples, ComDeBufferSample
from comde.rl.utils.get_shape import get_obs_shape, get_action_dim

try:
	# Check memory used by replay buffer when possible
	import psutil
except ImportError:
	psutil = None


class BaseBuffer(ABC):
	"""
	Base class that represent a buffer (rollout or replay)

	:param buffer_size: Max number of element in the buffer
	:param observation_space: Observation space
	:param action_space: Action space
	:param n_envs: Number of parallel environments
	"""

	def __init__(
		self,
		buffer_size: int,
		observation_space: spaces.Space,
		action_space: spaces.Space,
		n_envs: int = 1
	):
		super(BaseBuffer, self).__init__()
		self.buffer_size = buffer_size
		self.observation_space = observation_space
		self.action_space = action_space
		self.obs_shape = get_obs_shape(observation_space)

		self.action_dim = get_action_dim(action_space)
		self.pos = 0
		self.full = False
		self.n_envs = n_envs

	def __len__(self):
		raise NotImplementedError()

	@staticmethod
	def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
		"""
		Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
		to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
		to [n_steps * n_envs, ...] (which maintain the order)

		:param arr:
		:return:
		"""
		shape = arr.shape
		if len(shape) < 3:
			shape = shape + (1,)
		return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

	def size(self) -> int:
		"""
		:return: The current size of the buffer
		"""
		if self.full:
			return self.buffer_size
		return self.pos

	def add(self, *args, **kwargs) -> Tuple[int, int]:
		"""
		Add elements to the buffer.
		"""
		raise NotImplementedError()

	def extend(self, *args, **kwargs) -> None:
		"""
		Add a new batch of transitions to the buffer
		"""
		# Do a for loop along the batch axis
		for data in zip(*args):
			self.add(*data)

	def reset(self) -> None:
		"""
		Reset the buffer.
		"""
		self.pos = 0
		self.full = False

	def sample(
		self,
		batch_size: int = None,
		env: Optional[VecNormalize] = None,
		batch_inds: np.ndarray = None,
		get_batch_inds: bool = False
	) -> ComDeBufferSample:
		"""
		:param batch_size: Number of element to sample
		:param env: associated gym VecEnv
			to normalize the observations/rewards when sampling
		:param batch_inds
		:param get_batch_inds
		:return:
		"""
		upper_bound = self.buffer_size if self.full else self.pos
		if batch_inds is None:
			batch_inds = np.random.randint(0, upper_bound, size=batch_size)
		env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

		return self._get_samples(batch_inds, env_inds=env_inds, env=env, get_batch_inds=get_batch_inds)

	@abstractmethod
	def _get_samples(
		self,
		batch_inds: np.ndarray,
		env_inds: np.ndarray,
		env: Optional[VecNormalize] = None,
		get_batch_inds: bool = False
	) -> Union[ReplayBufferSamples]:
		raise NotImplementedError()

	@staticmethod
	def _normalize_obs(
		obs: Union[np.ndarray, Dict[str, np.ndarray]],
		env: Optional[VecNormalize] = None,
	) -> Union[np.ndarray, Dict[str, np.ndarray]]:
		if env is not None:
			return env.normalize_obs(obs)
		return obs

	@staticmethod
	def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
		if env is not None:
			return env.normalize_reward(reward).astype(np.float32)
		return reward
