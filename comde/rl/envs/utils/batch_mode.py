from contextlib import contextmanager
import gym
import numpy as np


class BatchEnv(gym.Wrapper):
	"""
		Return a batch shape of observations: [1, observation_dim]
		This is required for prediction
	"""
	def __init__(self, env: gym.Env):
		super(BatchEnv, self).__init__(env)
		self._batch_mode = False

	@contextmanager
	def batch_mode(self):
		prev_mode = self._batch_mode
		self._batch_mode = True
		yield
		self._batch_mode = prev_mode

	def step(self, action):
		observation, reward, done, info = super(BatchEnv, self).step(action)
		if self._batch_mode:
			observation = observation[np.newaxis, ...]
		return observation, reward, done, info

	def reset(self, **kwargs):
		observation = super(BatchEnv, self).reset(**kwargs)
		if self._batch_mode:
			observation = observation[np.newaxis, ...]
		return observation
