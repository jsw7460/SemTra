from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from comde.rl.buffers.base import BaseBuffer
from comde.rl.buffers.type_aliases import ComDeBufferSample


class SkillEpisodicMaskingBuffer(BaseBuffer):
	BUFFER_COMPONENTS = \
		{
			"observations", "next_observations", "actions", "dones",
			"first_observations", "skills", "skills_done",
		}

	def __init__(
		self,
		observation_space: gym.spaces.Space,
		action_space: gym.spaces.Space,
		subseq_len: int,
		n_envs: int = 1,
		buffer_size: int = -1,  # No matter
		use_all_previous_components: bool = False
	):
		"""
		This is a buffer for an episode.
		Thus sampled buffer has a shape [batch_size, max_len, (obs, act, reward, done, ...)-dim]
		"""
		super(SkillEpisodicMaskingBuffer, self).__init__(
			buffer_size=buffer_size,
			observation_space=observation_space,
			action_space=action_space,
			n_envs=n_envs
		)
		self.subseq_len = subseq_len
		self.episodes = []  # type: List[SkillContainedEpisode]
		self.episode_lengths = []
		self.min_episode_length = None
		self.max_episode_length = None
		self.representative_to_indices = defaultdict(list)
		self.use_all_previous_components = use_all_previous_components

	def __len__(self):
		return len(self.episodes)

	def get_mask(self, masking_size: int, nonmasking_size: int):
		raise DeprecationWarning("Obsolete")

	def add(self, episode: SkillContainedEpisode) -> None:
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
		infos = dataset["infos"]
		first_observations = dataset["first_observations"]
		skills = dataset["skills"]
		skills_done = dataset["skills_done"]
		if skills_done.ndim == 2:
			skills_done = np.squeeze(skills_done, axis=-1)
		skills_idxs = dataset["skills_idxs"]
		timestep = 0

		new_episode = SkillContainedEpisode()

		for i in range(n_data):
			new_episode.add(
				observation=observations[i],
				next_observation=next_observations[i],
				action=actions[i],
				reward=rewards[i],
				done=dones[i],
				info=infos[i],
				first_observation=first_observations[i],
				skill=skills[i],
				skill_done=skills_done[i],
				skill_idx=skills_idxs[i],
				timestep=timestep
			)
			timestep += 1
			if dones[i]:
				new_episode.set_rtgs_by_n_skills()
				new_episode.set_zeropaddings(self.subseq_len)
				if representative is not None:
					extracted_representative = new_episode.infos[0][representative][0]
					new_episode.representative = extracted_representative
					self.representative_to_indices[extracted_representative].append(self.pos)

				self.episodes.append(new_episode)
				self.episode_lengths.append(len(new_episode))
				self.pos += 1

				if clear_info:
					new_episode.clear_info()

				new_episode = SkillContainedEpisode()
				timestep = 0

		self.min_episode_length = min(self.episode_lengths)
		self.max_episode_length = max(self.episode_lengths)

	def _get_samples(
		self,
		batch_inds: np.ndarray,
		env_inds: np.ndarray,
		env: Optional[VecNormalize] = None,
		get_batch_inds: bool = False,
		give_noise: bool = False,
	) -> Union[Tuple, ComDeBufferSample]:

		episodes = [self.episodes[batch_idx] for batch_idx in batch_inds]
		threshes = np.array([len(episode) - self.subseq_len for episode in episodes])
		start_idxs = np.random.randint(0, threshes)
		observations = []
		next_observations = []
		actions = []
		dones = []
		first_observations = []
		skills = []
		skills_done = []
		maskings = []
		true_subseq_len = []
		timesteps = []

		# s = time.time()
		for episode, start_idx in zip(episodes, start_idxs):
			# for starting_idx in starting_idxs:
			# start_idx = np.random.randint(0, len(episode) - self.subseq_len)
			subtraj = episode.get_numpy_subtrajectory(from_=start_idx, to_=start_idx + self.subseq_len,
													  batch_mode=False)

			observations.append(subtraj["observations"].copy())
			next_observations.append(subtraj["next_observations"].copy())
			actions.append(subtraj["actions"].copy())
			dones.append(subtraj["dones"].copy())
			first_observations.append(subtraj["first_observations"].copy())
			skills.append(subtraj["skills"].copy())
			skills_done.append(subtraj["skills_done"].copy())
			maskings.append(subtraj["maskings"].copy())
			true_subseq_len.append(np.sum(subtraj["maskings"], keepdims=False))
			timesteps.append(np.arange(start_idx, start_idx + self.subseq_len))

		buffer_sample = ComDeBufferSample(
			observations=np.array(observations),
			next_observations=np.array(next_observations),
			actions=np.array(actions),
			dones=np.array(dones),
			first_observations=np.array(first_observations),
			skills=np.array(skills),
			skills_done=np.array(skills_done),
			maskings=np.array(maskings),
			true_subseq_len=np.array(true_subseq_len, dtype=np.int),
			timesteps=np.array(timesteps)
		)

		if get_batch_inds:
			return buffer_sample, batch_inds
		else:
			return buffer_sample

	def get_samples_from_timestep(self, batch_inds: np.ndarray, timesteps: np.ndarray, nearby_thresh: int):
		episodes = [self.episodes[batch_idx] for batch_idx in batch_inds]
		episode_lengths = [len(episode) for episode in episodes]
		start_idxs = [
			min(timestep[0] + nearby_thresh, episode_length - self.subseq_len - 1)
			for (timestep, episode_length) in zip(timesteps, episode_lengths)
		]
		observations = []
		next_observations = []
		actions = []
		dones = []
		first_observations = []
		skills = []
		skills_done = []
		maskings = []
		true_subseq_len = []
		timesteps = []

		# s = time.time()
		for episode, start_idx in zip(episodes, start_idxs):
			# for starting_idx in starting_idxs:
			# start_idx = np.random.randint(0, len(episode) - self.subseq_len)
			subtraj = episode.get_numpy_subtrajectory(
				from_=start_idx,
				to_=start_idx + self.subseq_len,
				batch_mode=False
			)

			observations.append(subtraj["observations"].copy())
			next_observations.append(subtraj["next_observations"].copy())
			actions.append(subtraj["actions"].copy())
			dones.append(subtraj["dones"].copy())
			first_observations.append(subtraj["first_observations"].copy())
			skills.append(subtraj["skills"].copy())
			skills_done.append(subtraj["skills_done"].copy())
			maskings.append(subtraj["maskings"].copy())
			true_subseq_len.append(np.sum(subtraj["maskings"], keepdims=False))
			timesteps.append(np.arange(start_idx, start_idx + self.subseq_len))

		buffer_sample = ComDeBufferSample(
			observations=np.array(observations),
			next_observations=np.array(next_observations),
			actions=np.array(actions),
			dones=np.array(dones),
			first_observations=np.array(first_observations),
			skills=np.array(skills),
			skills_done=np.array(skills_done),
			maskings=np.array(maskings),
			true_subseq_len=np.array(true_subseq_len, dtype=np.int),
			timesteps=np.array(timesteps)
		)

		return buffer_sample
