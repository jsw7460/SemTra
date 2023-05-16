"""Common aliases for type hints"""

from typing import NamedTuple, List, Union, Dict

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

	b: batch size
	l: subsequence length
	d: dimension
	M: Maximum possible number of skill in episode

	"""

	# === Task Agnostic Data ===
	observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	actions: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	next_observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]

	# === ComDe ===
	first_observations: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	# Note: M := The maximum possible number of skills in a trajectory
	source_skills: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b, M, d]
	source_skills_idxs: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b, M]
	target_skills: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b, M, d]
	target_skills_idxs: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b, M]
	n_source_skills: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b,]
	n_target_skills: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b,]
	sequential_requirement: Union[np.ndarray, th.tensor] = np.empty(0, )	# [b, d]
	str_sequential_requirement: List = []		# String
	non_functionality: Union[np.ndarray, th.tensor] = np.empty(0, )	# [b, d]
	source_parameters: List[Dict] = []
	parameters: List[Dict] = []

	skills: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l, d]
	skills_order: Union[np.ndarray, th.Tensor] = np.empty(0, )	# [b, l]
	skills_idxs: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	skills_done: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]

	# intents: Union[np.ndarray, th.Tensor] = None  # [b, l, d]
	params_for_skills: Union[np.ndarray, th.Tensor] = None  # [b, l, d]
	parameterized_skills: Union[np.ndarray, th.Tensor] = None  # [b, l, d], d = skill + non_functionality + param

	# === Transformer, ... ===
	rewards: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	dones: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	infos: List = np.empty(0, )  # [b, l]
	maskings: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	timesteps: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	rtgs: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, l]
	true_subseq_len: Union[np.ndarray, th.Tensor] = np.empty(0, )  # [b, ]

	# Maybe required (for some baseline which requires source state and actions)
	source_observations: Union[np.ndarray, th.Tensor] = np.empty(0,)
	source_actions: Union[np.ndarray, th.Tensor] = np.empty(0,)
	source_maskings: Union[np.ndarray, th.Tensor] = np.empty(0,)

	# Maybe required (for some baseline which requires video or their embeddings)
	source_videos: Union[np.ndarray, th.Tensor] = np.empty(0, )
	target_video: Union[np.ndarray, th.Tensor] = np.empty(0, )
	source_video_embeddings: Union[np.ndarray, th.Tensor] = np.empty(0,)
	target_video_embedding: Union[np.ndarray, th.Tensor] = np.empty(0,)


	def __repr__(self):
		for key in self._fields:
			print(f"{key} has shape {getattr(key).shape}")
		# base = f"observations: {self.observations.shape}\n" \
		# 	   f"actions: {self.actions.shape}\n" \
		# 	   f"first_observations: {self.first_observations.shape}\n" \
		# 	   f"skills: {self.skills.shape}\n" \
		# 	   f"skills_idxs: {self.skills_idxs.shape}\n" \
		# 	   f"skills_done: {self.skills_done.shape}\n"
		#
		# if self.maskings is None:
		# 	return base
		# else:
		# 	return base + f"next_observations: {self.next_observations.shape}\n" \
		# 				  f"rewards: {self.rewards.shape}\n" \
		# 				  f"dones: {self.dones.shape}\n" \
		# 				  f"infos: {len(self.infos)}\n" \
		# 				  f"maskings: {self.maskings.shape}\n" \
		# 				  f"timesteps: {self.timesteps.shape}\n" \
		# 				  f"rtgs: {self.rtgs.shape}"

	def __getitem__(self, idx: Union[slice, int]) -> "ComDeBufferSample":
		raise NotImplementedError("Obsolete")
		"""Slice for timesteps, not batch"""
		observations = self.observations[:, idx, ...]
		actions = self.actions[:, idx, ...]
		first_observations = self.first_observations[:, idx, ...]

		source_skills = self.source_skills.copy()
		target_skills = self.target_skills.copy()
		n_source_skills = self.n_source_skills.copy()
		n_target_skills = self.n_target_skills.copy()
		language_operators = self.language_operators.copy()

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
