from typing import List, Dict

import numpy as np


class Episode:
	"""
		To support a variable length episode, we use a list, not a numpy with predefined length
	"""

	def __init__(self):
		self.observations = []
		self.next_observations = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.infos = []
		self.maskings = []
		self.rtgs = []

	def __len__(self):
		return len(self.observations)

	def __getitem__(self, idx: slice):
		if idx.start < 0:
			idx = slice(0, idx.stop, None)
		observations = list(self.observations[idx].copy())
		next_observations = list(self.next_observations[idx].copy())
		actions = list(self.actions[idx].copy())
		rewards = list(self.rewards[idx].copy())
		dones = list(self.dones[idx].copy())
		infos = list(self.infos[idx].copy())
		maskings = list(self.maskings[idx].copy())
		rtgs = list(self.rtgs[idx].copy())
		return Episode.from_list(observations, next_observations, actions, rewards, dones, infos, maskings, rtgs)

	@staticmethod
	def expand_1st_dim(dataset: Dict[str, np.ndarray]):
		raise NotImplementedError("Out of date")
		# for k in dataset:
		# 	dataset[k] = np.expand_dims(dataset, axis=0)

	def get_numpy_subtrajectory(self, from_: int, to_: int, batch_mode: bool) -> Dict:
		assert from_ >= 0 and to_ < len(self)
		# TODO: We discard the info

		data = {
			"observations": np.array(self.observations[from_: to_]),
			"next_observations": np.array(self.next_observations[from_: to_]),
			"actions": np.array(self.actions[from_: to_]),
			"rewards": np.array(self.rewards[from_: to_]),
			"dones": np.array(self.dones[from_: to_]),
			"maskings": np.array(self.maskings[from_: to_])
		}

		if batch_mode:
			self.expand_1st_dim(data)
		return data

	def clear_info(self):
		self.infos = [[] for _ in range(len(self.infos))]

	@property
	# Note: These will raise an error if buffer is empty
	def observation_dim(self):
		return self.observations[0].shape[-1]

	def set_rtgs(self, gamma: float = 1.0) -> None:
		assert len(self) > 0
		discounted_cumsum = self.discount_cumsum(np.array(self.rewards), gamma)
		self.rtgs = discounted_cumsum.tolist()

	@staticmethod
	def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
		discount_cumsum = np.zeros_like(x)
		discount_cumsum[-1] = x[-1]
		for t in reversed(range(x.shape[0] - 1)):
			discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
		return discount_cumsum

	@staticmethod
	def from_list(
		observations: List,
		next_observations: List,
		actions: List,
		rewards: List,
		dones: List,
		infos: List,
		maskings: List,
		rtgs: List
	) -> "Episode":
		ret = Episode()
		ret.observations = observations
		ret.next_observations = next_observations
		ret.actions = actions
		ret.rewards = rewards
		ret.dones = dones
		ret.infos = infos
		ret.maskings = maskings
		ret.rtgs = rtgs
		return ret

	@property
	def action_dim(self):
		return self.actions[0].shape[-1]

	def get_return_to_go_at(self, timestep: int):
		return sum(self.rewards[timestep:])

	def add(
		self,
		observation: np.ndarray,
		next_observation: np.ndarray,
		action: np.ndarray,
		reward: np.ndarray,
		done: np.ndarray,
		info: List,
	):
		self.observations.append(observation.copy())
		self.next_observations.append(next_observation.copy())
		self.actions.append(action.copy())
		self.rewards.append(reward.copy())
		self.dones.append(done.copy())
		self.infos.append(info)
		self.maskings.append(np.array(1))  # Real data -> 1 // Padding data -> 0

	def to_numpydict(self) -> Dict:
		# shape: [episode_length, dim]
		data = {
			"observations": np.array(self.observations.copy()),
			"next_observations": np.array(self.actions.copy()),
			"actions": np.array(self.actions.copy()),
			"rewards": np.array(self.rewards.copy()),
			"dones": np.array(self.dones.copy()),
			"infos": self.infos.copy(),
			"maskings": self.maskings.copy(),
			"rtgs": np.array(self.rtgs.copy())
		}
		return data

	def set_zeropaddings(self, n_padding: int):
		for i in range(n_padding):
			self.observations.append(np.zeros(self.observation_dim, ))
			self.next_observations.append(np.zeros(self.observation_dim, ))
			self.actions.append(np.zeros(self.action_dim, ))
			self.rewards.append(np.array(0))
			self.dones.append(np.array(True))
			self.infos.append([])
			self.maskings.append(np.array(0))
			self.rtgs.append(0)

