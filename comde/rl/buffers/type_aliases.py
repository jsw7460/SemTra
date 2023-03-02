"""Common aliases for type hints"""

from typing import NamedTuple, List, Union

import numpy as np
import torch as th


class ReplayBufferSamples(NamedTuple):
	observations: np.ndarray
	actions: np.ndarray
	next_observations: np.ndarray
	dones: np.ndarray
	rewards: np.ndarray


class ComDeBufferSample(NamedTuple):
	"""
		This is for the "EPISODIC" buffer sample.
		So each component has shape [batch_size, 'subseq_len', dimension].
		This may contain unwanted zeropadding.

	b: batch size
	l: subsequence length
	d: dimension

	T: Variable integer. Can be sliced by timestep 0 to T
	"""
	observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	actions: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	first_observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	skills: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	skills_idxs: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	skills_done: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]

	# Followings are for the torch_transformer decoder model.
	next_observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	rewards: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	dones: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	infos: List = np.empty(0, )  # [b, l]
	maskings: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	timesteps: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	rtgs: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	true_subseq_len: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, ]

	# Followings are for the dynamic encoder which inputs the subsequence 'from first timestep to current'
	all_previous_observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, T, d]
	all_previous_actions: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, T, d]
	previous_observations_mean: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, d]
	previous_actions_mean: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b, d]

	def __repr__(self):
		"""Print helper"""
		base = f"observations: {self.observations.shape}\n" \
			   f"actions: {self.actions.shape}\n" \
			   f"first_observations: {self.first_observations.shape}\n" \
			   f"skills: {self.skills.shape}\n" \
			   f"skills_idxs: {self.skills_idxs.shape}\n" \
			   f"skills_done: {self.skills_done.shape}\n"

		if self.maskings is None:
			return base
		else:
			return base + f"next_observations: {self.next_observations.shape}\n" \
						  f"rewards: {self.rewards.shape}\n" \
						  f"dones: {self.dones.shape}\n" \
						  f"infos: {len(self.infos)}\n" \
						  f"maskings: {self.maskings.shape}\n" \
						  f"timesteps: {self.timesteps.shape}\n" \
						  f"rtgs: {self.rtgs.shape}"

	def to_tensor(self, device: th.device) -> "ComDeBufferSample":
		raise NotImplementedError()

	# tensorized_sample = {}
	# for key, value in self.__dict__.items():
	# 	if value is not None and key != "infos":
	# 		value = th.tensor(value, device=device, dtype=th.float32)
	# 	tensorized_sample[key] = value
	# return MMSbrlOfflineBufferSample(**tensorized_sample)

	def flatten_basic_components(self) -> "ComDeBufferSample":
		raise NotImplementedError("Obsolete")

	# """
	# Since next_observations ~ rtgs are used only for torch_transformer model,
	# flatten is only required for basic components (observations ~ skills_done)
	# """
	# return MMSbrlOfflineBufferSample(
	# 	observations=self.observations.reshape(-1, self.observations.shape[-1]),
	# 	actions=self.actions.reshape(-1, self.actions.shape[-1]),
	# 	first_observations=self.first_observations.reshape(-1, self.first_observations.shape[-1]),
	# 	skills=self.skills.reshape(-1, self.skills.shape[-1]),
	# 	skills_idxs=self.skills_idxs.reshape(-1, ),
	# 	skills_done=self.skills_done.reshape(-1, )
	# )

	def get_last_timestep_components(self) -> "ComDeBufferSample":
		data = {
			"observations": self.observations[:, -1, ...],
			"actions": self.actions[:, -1, ...],
			"first_observations": self.first_observations[:, -1, ...],
			"skills": self.skills[:, -1],
			# "skills_idxs": self.skills_idxs[:, -1],
			"skills_done": self.skills_done[:, -1]
		}
		if self.maskings is None:
			return ComDeBufferSample(**data)
		else:
			data.update(
				{
					"next_observations": self.next_observations[:, -1, ...],
					# "rewards": self.rewards[:, -1],
					"dones": self.dones[:, -1],
					# "infos": self.infos[:, -1],
					"maskings": self.maskings[:, -1],
					# "timesteps": self.timesteps[:, -1],
					# "rtgs": self.rtgs[:, -1],
					"true_subseq_len": self.true_subseq_len[-1]
				}
			)
			return ComDeBufferSample(**data)

	def __getitem__(self, idx: Union[slice, int]) -> "ComDeBufferSample":
		"""Slice for timesteps, not batch"""
		observations = self.observations[:, idx, ...]
		actions = self.actions[:, idx, ...]
		first_observations = self.first_observations[:, idx, ...]
		skills = self.skills[:, idx, ...]
		skills_idxs = self.skills_idxs[:, idx]
		skills_done = self.skills_done[:, idx]
		next_observations = self.next_observations[:, idx, ...]
		rewards = self.rewards[:, idx]
		dones = self.dones[:, idx]
		infos = self.infos[:, idx]  # This is dummy
		maskings = self.maskings[:, idx]
		timesteps = self.timesteps[:, idx]
		rtgs = self.rtgs[:, idx]
		true_subseq_len = self.true_subseq_len.copy()

		return ComDeBufferSample(
			observations=observations,
			actions=actions,
			first_observations=first_observations,
			skills=skills,
			skills_idxs=skills_idxs,
			skills_done=skills_done,
			next_observations=next_observations,
			rewards=rewards,
			dones=dones,
			infos=infos,
			maskings=maskings,
			timesteps=timesteps,
			rtgs=rtgs,
			true_subseq_len=true_subseq_len
		)

	def get_batch_from_idx(self, idx: Union[slice, int]) -> "ComDeBufferSample":
		return ComDeBufferSample(
			observations=self.observations[idx, ...],
			actions=self.actions[idx, ...],
			first_observations=self.first_observations[idx, ...],
			skills=self.skills[idx, ...],
			skills_idxs=self.skills_idxs[idx, ...],
			skills_done=self.skills_done[idx, ...],
			next_observations=self.next_observations[idx, ...],
			rewards=self.rewards[idx, ...],
			dones=self.dones[idx, ...],
			infos=self.infos[idx],
			maskings=self.maskings[idx, ...],
			timesteps=self.timesteps[idx, ...],
			rtgs=self.rtgs[idx, ...],
			true_subseq_len=self.true_subseq_len[idx]
		)
