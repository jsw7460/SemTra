from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from comde.rl.buffers.buffers.base_buffer import BaseBuffer
from comde.rl.buffers.episodes.base import Episode
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.rl.envs.base import ComdeSkillEnv


class EpisodicMaskingBuffer(BaseBuffer):
	BUFFER_COMPONENTS = \
		{
			"observations", "next_observations", "actions", "rewards", "dones"
		}

	def __init__(
		self,
		env: ComdeSkillEnv,
		subseq_len: int,
		n_envs: int = 1,
		buffer_size: int = -1,  # No matter
		use_all_previous_components: bool = False
	):
		"""
		This is a buffer for an episode.
		Thus sampled buffer has a shape [batch_size, max_len, (obs, act, reward, done, ...)-dim]
		"""
		super(EpisodicMaskingBuffer, self).__init__(
			buffer_size=buffer_size,
			env=env,
			n_envs=n_envs
		)
		self.subseq_len = subseq_len
		self.episodes = []  # type: List[Episode]
		self.episode_lengths = []
		self.representative_to_indices = defaultdict(list)
		self.use_all_previous_components = use_all_previous_components

		self.min_episode_length = None
		self.max_episode_length = None

	def __len__(self):
		return len(self.episodes)

	def get_mask(self, masking_size: int, nonmasking_size: int):
		raise DeprecationWarning("No usage. Masking ")

	def add(self, episode: Episode) -> None:
		assert len(episode) > self.subseq_len, \
			"Too short episode. Please remove this episode or decrease the subseq len."
		self.episodes.append(episode)
		self.pos += 1

	def add_dict_chunk(self, dataset: Dict, representative: str = None, clear_info: bool = False) -> None:
		"""
			save d4rl style dataset(dictionary)
			if clear_info: We clear all the info list. This can save memory in a certain dataset.
		"""
		if dataset.get("terminals", None) is not None:
			dataset["dones"] = dataset["terminals"]

		if dataset.get("next_observations", None) is None:
			next_observations = np.zeros_like(dataset["observations"])
			next_observations[: -1] = dataset["observations"][1:]
			dataset["next_observations"] = next_observations

		assert self.BUFFER_COMPONENTS <= dataset.keys(), \
			f"Not every component required for define episode is not in dataset: " \
			f"Missing part is {self.BUFFER_COMPONENTS - dataset.keys()}"

		n_data = len(dataset["observations"])
		observations = dataset["observations"]
		next_observations = dataset["next_observations"]
		actions = dataset["actions"]
		rewards = dataset["rewards"]
		if rewards.ndim == 2:
			rewards = np.squeeze(rewards, axis=-1)
		dones = dataset["dones"]
		if dones.ndim == 2:
			dones = np.squeeze(dones, axis=-1)
		if "infos" not in dataset.keys():
			dataset["infos"] = [{} for _ in range(n_data)]
		infos = dataset["infos"]

		new_episode = Episode()

		for i in range(n_data):
			new_episode.add(
				observation=observations[i],
				next_observation=next_observations[i],
				action=actions[i],
				reward=rewards[i],
				done=dones[i],
				info=infos[i],
			)
			if dones[i]:
				new_episode.set_rtgs()
				new_episode.set_zeropaddings(self.subseq_len)
				self.episodes.append(new_episode)
				self.episode_lengths.append(len(new_episode))
				self.pos += 1

				if clear_info:
					new_episode.clear_info()

				new_episode = Episode()

		self.min_episode_length = min(self.episode_lengths)
		self.max_episode_length = max(self.episode_lengths)

	def _get_samples(
		self,
		batch_inds: np.ndarray,
		env_inds: np.ndarray,
		env: Optional[VecNormalize] = None,
		get_batch_inds: bool = False
	) -> Union[Tuple, ComDeBufferSample]:

		subtrajectories = []

		all_prev_observations = []
		all_prev_actions = []
		timesteps = []

		episodes = [self.episodes[batch_idx] for batch_idx in batch_inds]
		# starting_idxs = [np.random.randint(0, len(episode) - self.subseq_len) for episode in episodes]

		for episode in episodes:
			# for starting_idx in starting_idxs:
			starting_idx = np.random.randint(0, len(episode) - self.subseq_len)
			subtrajectory = episode[starting_idx: starting_idx + self.subseq_len]
			subtrajectories.append(subtrajectory)
			timesteps.append(np.arange(starting_idx, starting_idx + self.subseq_len))

			if self.use_all_previous_components:
				starting_idx_for_all_previous = np.random.randint(0, self.min_episode_length)
				all_prev_observations.append(np.array(episode.observations[:starting_idx_for_all_previous]))
				all_prev_actions.append(np.array(episode.actions[:starting_idx_for_all_previous]))

		numpydict_subtrajectories = [subtraj.to_numpydict() for subtraj in subtrajectories]
		data = dict()
		training_buffer_components = list(EpisodicMaskingBuffer.BUFFER_COMPONENTS) + ["maskings", "rtgs"]

		for key in training_buffer_components:
			data[key] = np.stack(
				[numpydict_subtrajectory[key] for numpydict_subtrajectory in numpydict_subtrajectories]
			)

		timesteps = np.array(timesteps)
		buffer_sample = ComDeBufferSample(**data)
		buffer_sample = buffer_sample._replace(true_subseq_len=np.sum(buffer_sample.maskings, axis=1))
		buffer_sample = buffer_sample._replace(timesteps=timesteps)

		if get_batch_inds:
			return buffer_sample, batch_inds
		else:
			return buffer_sample

	def get_samples_from_timestep(self, batch_inds: np.ndarray, timesteps: np.ndarray, nearby_thresh: int):
		episodes = [self.episodes[batch_idx] for batch_idx in batch_inds]
		episode_lengths = [len(episode) for episode in episodes]
		subtrajectories = []

		all_prev_observations = []
		all_prev_actions = []
		start_idxs = [
			min(timestep[0] + nearby_thresh, episode_length - self.subseq_len - 1)
			for (timestep, episode_length) in zip(timesteps, episode_lengths)
		]
		for episode, starting_idx in zip(episodes, start_idxs):
			# for starting_idx in starting_idxs:
			subtrajectory = episode[starting_idx: starting_idx + self.subseq_len]
			subtrajectories.append(subtrajectory)

			if self.use_all_previous_components:
				starting_idx_for_all_previous = np.random.randint(0, self.min_episode_length)
				all_prev_observations.append(np.array(episode.observations[:starting_idx_for_all_previous]))
				all_prev_actions.append(np.array(episode.actions[:starting_idx_for_all_previous]))

		numpydict_subtrajectories = [subtraj.to_numpydict() for subtraj in subtrajectories]
		data = dict()
		training_buffer_components = list(EpisodicMaskingBuffer.BUFFER_COMPONENTS) + ["maskings"]
		for key in training_buffer_components:
			data[key] = np.stack(
				[numpydict_subtrajectory[key] for numpydict_subtrajectory in numpydict_subtrajectories]
			)

		buffer_sample = ComDeBufferSample(**data)
		buffer_sample = buffer_sample._replace(true_subseq_len=np.sum(buffer_sample.maskings, axis=1))
		return buffer_sample
