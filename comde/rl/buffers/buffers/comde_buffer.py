import pickle
import random
from typing import Dict, Optional, Union, Tuple, List

import gym
import h5py
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from comde.rl.buffers.buffers.episodic import EpisodicMaskingBuffer
from comde.rl.buffers.episodes.source_target_skill import SourceTargetSkillContainedEpisode
from comde.rl.buffers.type_aliases import ComDeBufferSample


class ComdeBuffer(EpisodicMaskingBuffer):
	MUST_LOADED_COMPONENTS = {
		"observations", "actions", "skills_idxs", "skills_order", "skills_done", "source_skills", "target_skills"
	}

	def __init__(
		self,
		observation_space: gym.spaces.Space,
		action_space: gym.spaces.Space,
		subseq_len: int,
		n_envs: int = 1,
		buffer_size: int = -1,  # No matter
	):
		super(ComdeBuffer, self).__init__(
			observation_space=observation_space,
			action_space=action_space,
			subseq_len=subseq_len,
			n_envs=n_envs,
			buffer_size=buffer_size,
			use_all_previous_components=False
		)
		del self.representative_to_indices

	def add_dict_chunk(self, dataset: Dict, representative: str = None, clear_info: bool = False) -> None:
		raise NotImplementedError("This is only for pickle file. ComDe does not support it.")

	def add_episodes_from_h5py(self, paths: Dict[str, Union[List, str]]):
		"""
			## README ##
			- Each path in paths corresponds to one trajectory.
			- "skills" are processed using "skills_idxs" when making minibatch. So we add 'None' skill to buffer.
		"""
		trajectory_paths = paths["trajectory"]
		language_guidance_paths = paths["language_guidance"]

		with open(language_guidance_paths, "rb") as f:
			language_guidance = pickle.load(f)

		ep: SourceTargetSkillContainedEpisode

		for path in trajectory_paths:
			episode = SourceTargetSkillContainedEpisode()
			trajectory = h5py.File(path, "r")
			dataset = self.preprocess_h5py_trajectory(trajectory, language_guidance)
			episode.add_from_dict(dataset)

			self.add(episode)
			self.episode_lengths.append(len(episode))

		self.min_episode_length = min(self.episode_lengths)
		self.max_episode_length = max(self.episode_lengths)
		max_source = max([ep.n_source_skills for ep in self.episodes])
		max_target = max([ep.n_target_skills for ep in self.episodes])
		max_possible_skills = max(max_source, max_target)

		[ep.set_zeropaddings(
			n_padding=self.subseq_len,
			max_possible_skills=max_possible_skills
		) for ep in self.episodes]

	def preprocess_h5py_trajectory(
		self,
		trajectory: h5py.File,
		language_guidance_mapping: Dict[str, Dict[str,  np.ndarray]]
	) -> Dict:
		assert self.MUST_LOADED_COMPONENTS <= trajectory.keys(), \
			f"Under qualified dataset. Please fill {trajectory.keys() - self.MUST_LOADED_COMPONENTS}"

		observations = np.array(trajectory["observations"])
		next_observations = np.zeros_like(observations)
		next_observations[: -1] = observations[1:]
		actions = np.array(trajectory["actions"])

		traj_len = len(observations)
		rewards = np.zeros((traj_len,))
		rtgs = np.zeros((traj_len,))
		dones = np.zeros_like(rewards, dtype=np.bool)

		if "infos" in trajectory.keys():
			assert type(trajectory["infos"]) == List, "undefined info type"
			infos = list(trajectory["infos"])
		else:
			infos = [[] for _ in range(traj_len)]

		source_skills = []
		for skills_in_demo in trajectory["source_skills"].values():
			for skill in skills_in_demo:
				source_skills.append(skill)

		language_guidance_vectors = language_guidance_mapping[
			str(trajectory["operator"][()], "utf-8")	# sequential, parallel, ...
		].values()
		language_guidance_vector = random.choice(list(language_guidance_vectors))

		# === Compute first observations ===
		skills_idxs = np.array(trajectory["skills_idxs"])
		first_observations = np.zeros_like(observations)
		first_observations[0] = observations[0]
		first_obs_pos = 0
		for i in range(1, traj_len):
			if skills_idxs[i - 1] != skills_idxs[i]:
				first_obs_pos = i
			first_observations[i] = observations[first_obs_pos]

		# === Augment skill done by 4 ===
		n_aug = 4
		done_times = np.where(trajectory["skills_done"] == 1)[0]
		augmented_skills_done = np.array(trajectory["skills_done"]).copy()
		for timestep in done_times.astype("i4"):
			augmented_skills_done[timestep - n_aug: timestep + n_aug + 1] = 1

		dataset = {
			"observations": observations,
			"next_observations": next_observations,
			"actions": actions,
			"rewards": rewards,
			"dones": dones,
			"infos": infos,
			"source_skills": source_skills,
			"target_skills": list(trajectory["target_skills"]),
			"language_operator": language_guidance_vector,
			"first_observations": first_observations,
			"skills_done": augmented_skills_done,
			"skills_order": np.array(trajectory["skills_order"]),
			"skills_idxs": skills_idxs,
			"rtgs": rtgs
		}

		return dataset

	def _get_samples(
		self,
		batch_inds: np.ndarray,
		env_inds: np.ndarray,
		env: Optional[VecNormalize] = None,
		get_batch_inds: bool = False
	) -> Union[Tuple, ComDeBufferSample]:
		episodes = [self.episodes[batch_idx] for batch_idx in batch_inds]
		threshes = np.array([len(episode) - self.subseq_len for episode in episodes])
		start_idxs = np.random.randint(0, threshes)

		# === Task Agnostic Data ===
		observations = []
		actions = []
		next_observations = []

		# === ComDe ===
		first_observations = []
		source_skills = []
		target_skills = []
		n_source_skills = []
		n_target_skills = []
		language_operators = []
		skills = []
		skills_order = []
		skills_idxs = []
		skills_done = []

		# === Transformer, ... ===
		maskings = []
		timesteps = []

		for ep, start_idx in zip(episodes, start_idxs):
			subtraj = ep.get_numpy_subtrajectory(from_=start_idx, to_=start_idx + self.subseq_len, batch_mode=False)
			observations.append(subtraj["observations"])
			actions.append(subtraj["actions"])
			next_observations.append(subtraj["next_observations"])

			first_observations.append(subtraj["first_observations"])
			source_skills.append(subtraj["source_skills"])
			target_skills.append(subtraj["target_skills"])
			n_source_skills.append(subtraj["n_source_skills"])
			n_target_skills.append(subtraj["n_target_skills"])
			language_operators.append(subtraj["language_operator"])
			skills.append(subtraj["skills"])
			skills_order.append(subtraj["skills_order"])
			skills_idxs.append(subtraj["skills_idxs"])

			skills_done.append(subtraj["skills_done"])
			maskings.append(subtraj["maskings"])
			timesteps.append(np.arange(start_idx, start_idx + self.subseq_len))

		buffer_sample = ComDeBufferSample(
			observations=np.array(observations),
			actions=np.array(actions),
			next_observations=np.array(next_observations),
			first_observations=np.array(first_observations),
			source_skills=np.array(source_skills),
			target_skills=np.array(target_skills),
			n_source_skills=np.array(n_source_skills),
			n_target_skills=np.array(n_target_skills),
			language_operators=np.array(language_operators),
			skills=np.array(skills),
			skills_order=np.array(skills_order).astype("i4"),
			skills_idxs=np.array(skills_idxs),
			skills_done=np.array(skills_done),
			maskings=np.array(maskings),
			timesteps=np.array(timesteps)
		)

		return buffer_sample
