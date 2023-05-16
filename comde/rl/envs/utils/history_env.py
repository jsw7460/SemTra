import collections
from typing import Dict, Tuple

import gym
import numpy as np


class WrappedGymEnv:

	def __getattr__(self, name):
		"""Wrappers forward non-overridden method calls to their wrapped env."""
		if name.startswith('__'):
			raise AttributeError(name)
		return getattr(self.env, name)


class HistoryEnv(gym.Wrapper):
	"""Environment wrapper for supporting sequential model inference.
	"""

	def __init__(
		self,
		env: gym.Env,
		num_stack_frames: int = 1
	):
		super(HistoryEnv, self).__init__(env=env)
		self.env = env

		self.num_stack_frames = num_stack_frames
		if self.is_goal_conditioned:
			# If env is goal-conditioned, we want to track goal history.
			self.goal_stack = collections.deque([], maxlen=self.num_stack_frames)
		self.obs_stack = collections.deque([], maxlen=self.num_stack_frames)
		self.act_stack = collections.deque([], maxlen=self.num_stack_frames)
		self.rew_stack = collections.deque([], maxlen=self.num_stack_frames)
		self.done_stack = collections.deque([], maxlen=self.num_stack_frames)
		self.info_stack = collections.deque([], maxlen=self.num_stack_frames)

	@property
	def observation_space(self):
		"""Constructs observation space."""
		parent_obs_space = self.env.observation_space
		act_space = self.action_space
		episode_history = {
			'observations': gym.spaces.Box(
				np.stack([parent_obs_space.low] * self.num_stack_frames, axis=0),
				np.stack([parent_obs_space.high] * self.num_stack_frames, axis=0),
				dtype=parent_obs_space.dtype),
			'actions': gym.spaces.Box(
				np.stack([act_space.low] * self.num_stack_frames, axis=0),
				np.stack([act_space.high] * self.num_stack_frames, axis=0),
				dtype=act_space.dtype
			),
			'rewards': gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames])
		}
		if self.is_goal_conditioned:
			goal_shape = np.shape(self.env.goal)  # pytype: disable=attribute-error
			episode_history['returns-to-go'] = gym.spaces.Box(
				-np.inf, np.inf, [self.num_stack_frames] + goal_shape)
		return gym.spaces.Dict(**episode_history)

	@property
	def is_goal_conditioned(self):
		return False

	def pad_current_episode(self, obs, n):
		# Prepad current episode with n steps.
		for _ in range(n):
			if self.is_goal_conditioned:
				self.goal_stack.append(self.env.goal)  # pytype: disable=attribute-error
			self.obs_stack.append(np.zeros_like(obs))
			self.act_stack.append(np.zeros_like(self.env.action_space.sample()))
			self.rew_stack.append(0)
			self.done_stack.append(1)
			self.info_stack.append(None)

	def _get_observation(self) -> Dict[str, np.ndarray]:
		"""Return current episode's N-stacked observation.

		For N=3, the first observation of the episode (reset) looks like:

		*= hasn't happened yet.

		GOAL  OBS  ACT  REW  DONE
		=========================
		g0    0    0.   0.   True
		g0    0    0.   0.   True
		g0    x0   0.   0.   False

		After the first step(a0) taken, yielding x1, r0, done0, info0, the next
		observation looks like:

		GOAL  OBS  ACT  REW  DONE
		=========================
		g0    0    0.   0.   True
		g0    x0   0.   0.   False
		g1    x1   a0   r0   d0

		A more chronologically intuitive way to re-order the column data would be:

		PREV_ACT  PREV_REW  PREV_DONE CURR_GOAL CURR_OBS
		================================================
		0.        0.        True      g0        0
		0.        0.        False*    g0        x0
		a0        r0        info0     g1        x1

		Returns:
		  episode_history: np.ndarray of observation.
		"""
		episode_history = {
			'observations': np.stack(self.obs_stack, axis=0),
			'actions': np.stack(self.act_stack, axis=0),
			'rewards': np.stack(self.rew_stack, axis=0),
		}
		if self.is_goal_conditioned:
			episode_history['returns-to-go'] = np.stack(self.goal_stack, axis=0)
		return episode_history

	def reset(self, *args, **kwargs):
		"""Resets env and returns new observation."""
		obs = self.env.reset()
		# Create a N-1 "done" past frames.
		self.pad_current_episode(obs, self.num_stack_frames - 1)
		# Create current frame (but with placeholder actions and rewards).
		if self.is_goal_conditioned:
			self.goal_stack.append(self.env.goal)
		self.obs_stack.append(obs)
		self.act_stack.append(np.zeros_like(self.env.action_space.sample()))
		self.rew_stack.append(0)
		self.done_stack.append(0)
		self.info_stack.append(None)
		ret = self._get_observation()
		return ret

	def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
		"""Replaces env observation with fixed length observation history."""
		# Update applied action to the previous timestep.

		store_action = action.copy()
		store_action[0] += 0.0

		self.act_stack[-1] = store_action
		obs, rew, done, info = self.env.step(np.array(action))
		self.rew_stack[-1] = rew
		# Update frame stack.
		self.obs_stack.append(obs)
		self.act_stack.append(np.zeros_like(self.env.action_space.sample()))  # Append unknown action to current timestep.
		self.rew_stack.append(0)
		self.info_stack.append(info)
		if self.is_goal_conditioned:
			self.goal_stack.append(self.env.goal)
		if done:
			if self.is_goal_conditioned:
				# rewrite the observations to reflect hindsight RtG conditioning.
				self.replace_goals_with_hindsight()

		return self._get_observation(), rew, done, info

	def replace_goals_with_hindsight(self):
		# We perform this after rew_stack has been updated.
		assert self.is_goal_conditioned
		window_return = sum(list(self.rew_stack))
		for r in self.rew_stack:
			self.goal_stack.append(window_return)
			window_return -= r


class SkillHistoryEnv(HistoryEnv):
	def __init__(self, env: gym.Env, skill_dim: int, num_stack_frames: int = 1):
		super(SkillHistoryEnv, self).__init__(env=env, num_stack_frames=num_stack_frames)
		self.skill_stack = collections.deque([], maxlen=self.num_stack_frames)
		self.skill_space = gym.spaces.Box(-np.inf, np.inf, shape=(skill_dim, ))
		self.skill_dim = skill_dim
		self.n_masking = num_stack_frames

	@property
	def observation_space(self):
		"""Constructs observation space."""
		parent_obs_space = self.env.observation_space
		act_space = self.action_space

		episode_history = {
			"observations": gym.spaces.Box(
				np.stack([parent_obs_space.low] * self.num_stack_frames, axis=0),
				np.stack([parent_obs_space.high] * self.num_stack_frames, axis=0),
				dtype=parent_obs_space.dtype),
			"actions": gym.spaces.Box(
				np.stack([act_space.low] * self.num_stack_frames, axis=0),
				np.stack([act_space.high] * self.num_stack_frames, axis=0),
				dtype=act_space.dtype
			),
			"skills": gym.spaces.Box(
				np.stack([self.skill_space.low] * self.num_stack_frames, axis=0),
				np.stack([self.skill_space.high] * self.num_stack_frames, axis=0)
			),
			"rewards": gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames])
		}
		if self.is_goal_conditioned:
			goal_shape = np.shape(self.env.goal)  # pytype: disable=attribute-error
			episode_history['returns-to-go'] = gym.spaces.Box(
				-np.inf, np.inf, [self.num_stack_frames] + goal_shape)
		return gym.spaces.Dict(**episode_history)

	def pad_current_episode(self, obs: np.ndarray, n: int):
		super(SkillHistoryEnv, self).pad_current_episode(obs=obs, n=n)
		for _ in range(n):
			self.skill_stack.append(np.zeros_like(self.skill_space.sample()))

	def _get_observation(self) -> Dict[str, np.ndarray]:
		ret = super(SkillHistoryEnv, self)._get_observation()
		ret.update({
			"skills": np.stack(self.skill_stack, axis=0),
			"maskings": np.concatenate(
				(np.zeros((self.n_masking,)), np.ones((self.num_stack_frames - self.n_masking)))
			)
		})
		return ret

	def reset(self, init_skill: np.ndarray):
		"""Resets env and returns new observation."""
		obs = self.env.reset()
		# Create a N-1 "done" past frames.
		self.pad_current_episode(obs, self.num_stack_frames - 1)
		# Create current frame (but with placeholder actions and rewards).
		if self.is_goal_conditioned:
			self.goal_stack.append(self.env.goal)
		self.obs_stack.append(obs)

		self.n_masking = self.num_stack_frames - 1

		self.act_stack.append(np.zeros_like(self.env.action_space.sample()))
		self.rew_stack.append(0)
		self.done_stack.append(0)
		self.info_stack.append(None)
		self.skill_stack.append(init_skill)
		ret = self._get_observation()
		return ret

	def step(self, action: np.ndarray, skill: np.ndarray = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
		self.skill_stack.append(skill)
		action = self.get_buffer_action(action)
		if self.n_masking > 0:
			self.n_masking -= 1
		return super(SkillHistoryEnv, self).step(action=action)