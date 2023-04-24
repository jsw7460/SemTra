import ast
import pickle
import random
from typing import Dict, Optional, Union, Tuple, List, Any

import gym
import h5py
import numpy as np
from jax.tree_util import tree_map
from stable_baselines3.common.vec_env import VecNormalize

from comde.rl.buffers.buffers.episodic import EpisodicMaskingBuffer
from comde.rl.buffers.episodes.source_target_skill import SourceTargetSkillContainedEpisode
from comde.rl.buffers.episodes.source_target_state import SourceStateEpisode
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
		cfg: Dict,
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

		self.num_skills_done_relabel = cfg["num_skills_done_relabel"]
		self.max_source_skills = cfg["max_source_skills"]
		self.max_target_skills = cfg["max_target_skills"]
		self.save_source_trajectory = cfg["save_source_trajectory"]
		self.default_source_skills = cfg["default_source_skills"]
		self.episode_class = SourceStateEpisode if self.save_source_trajectory else SourceTargetSkillContainedEpisode

	def add_dict_chunk(self, dataset: Dict, representative: str = None, clear_info: bool = False) -> None:
		raise NotImplementedError("This is only for pickle file. ComDe does not support it.")

	def add_episodes_from_h5py(self, paths: Dict[str, Union[List, str]]):
		"""
			## README ##
			- Each path in paths corresponds to one trajectory.
			- "skills" are processed using "skills_idxs" when making minibatch. So we add 'None' skill to buffer.
		"""
		trajectory_paths = paths["trajectory"]
		sequential_requirements_path = paths["sequential_requirements"]
		non_functionalities_path = paths["non_functionalities"]

		# num_skills_done_relabel = cfg["num_skills_done_relabel"]

		with open(sequential_requirements_path, "rb") as f:
			sequential_requirements_mapping = pickle.load(f)

		with open(non_functionalities_path, "rb") as f:
			non_functionalities_mapping = pickle.load(f)

		if self.save_source_trajectory:
			ep: SourceStateEpisode
		else:
			ep: SourceTargetSkillContainedEpisode

		for path in trajectory_paths:
			episode = self.episode_class()
			trajectory = h5py.File(path, "r")

			dataset = self.preprocess_h5py_trajectory(
				trajectory,
				sequential_requirements_mapping,
				non_functionalities_mapping,
				num_skills_done_relabel=self.num_skills_done_relabel,
			)
			if dataset is None:
				continue
			episode.add_from_dict(dataset)

			self.add(episode)
			self.episode_lengths.append(len(episode))

		self.min_episode_length = min(self.episode_lengths)
		self.max_episode_length = max(self.episode_lengths)

		[ep.set_zeropaddings(
			n_padding=self.subseq_len,
			max_source_skills=self.max_source_skills,
			max_target_skills=self.max_target_skills
		) for ep in self.episodes]

	def preprocess_h5py_trajectory(
		self,
		trajectory: h5py.File,
		sequential_requirements_mapping: Dict[str, Dict[str, np.ndarray]],
		non_functionalities_mapping: Dict[str, Dict[str, np.ndarray]],
		num_skills_done_relabel: int,
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
			for skill in np.array(skills_in_demo):
				source_skills.append(skill)
		target_skills = list(trajectory["target_skills"])

		sequential_requirement = str(trajectory["sequential_requirement"][()], "utf-8")
		non_functionality = str(trajectory["non_functionality"][()], "utf-8")
		skills_idxs = np.array(trajectory["skills_idxs"])
		parameter = str(trajectory["parameter"][()], "utf-8")
		parameter = ast.literal_eval(parameter)
		source_parameter = self.default_source_skills[non_functionality].copy()

		for key, value in parameter.items():  # Make into float value (because hdf5 save string).
			parameter[key] = float(value)

		params_for_skills = self.get_skills_parameter(
			skills_idxs=skills_idxs,
			parameter=parameter
		)
		sequential_requirement_vector = sequential_requirements_mapping[sequential_requirement]
		non_functionality_vector = non_functionalities_mapping[non_functionality]

		# === Compute first observations ===
		first_observations = np.zeros_like(observations)
		first_observations[0] = observations[0]
		first_obs_pos = 0
		for i in range(1, traj_len):
			if skills_idxs[i - 1] != skills_idxs[i]:
				first_obs_pos = i
			first_observations[i] = observations[first_obs_pos]

		# === Augment skill done by 4 ===
		done_times = np.where(np.array(trajectory["skills_done"]) == 1)[0]
		augmented_skills_done = np.array(trajectory["skills_done"]).copy()

		for timestep in done_times.astype("i4"):
			augmented_skills_done[timestep - num_skills_done_relabel: timestep + num_skills_done_relabel + 1] = 1

		dataset = {
			"observations": observations,
			"next_observations": next_observations,
			"actions": actions,
			"rewards": rewards,
			"dones": dones,
			"infos": infos,
			"source_skills_idxs": source_skills,
			"target_skills_idxs": target_skills,
			"sequential_requirement": sequential_requirement_vector,  # Sequential requirements act as an operator.
			"non_functionality": non_functionality_vector,
			"first_observations": first_observations,
			"skills_done": augmented_skills_done,
			"skills_order": np.array(trajectory["skills_order"]),
			"skills_idxs": skills_idxs,
			"params_for_skills": params_for_skills,
			"rtgs": rtgs,
			"source_parameter": source_parameter,
			"parameter": parameter,
		}
		if self.save_source_trajectory:
			source_observations = np.array(trajectory["source_observations"])
			source_actions = np.array(trajectory["source_actions"])
			dataset.update({"source_observations": source_observations, "source_actions": source_actions})
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

		# === ComDe Components ===
		source_skills = []
		target_skills = []
		n_source_skills = []
		n_target_skills = []
		sequential_requirements = []
		non_functionalities = []
		subtrajectories = []
		source_parameters = []
		parameters = []

		for ep, start_idx in zip(episodes, start_idxs):
			subtraj = ep.get_numpy_subtrajectory(from_=start_idx, to_=start_idx + self.subseq_len, batch_mode=False)
			subtraj.pop("rtgs")
			source_parameters.append(subtraj.pop("source_parameter"))
			parameters.append(subtraj.pop("parameter"))
			# source_observations.append(subtraj.pop("source_observations"))
			# source_actions.append(subtraj.pop("source_actions"))

			subtrajectories.append(subtraj)
			source_skills.append(subtraj.pop("source_skills"))
			target_skills.append(subtraj.pop("target_skills"))
			sequential_requirements.append(random.choice(list(subtraj.pop("sequential_requirement").values())))
			non_functionalities.append(random.choice(list(subtraj.pop("non_functionality").values())))
			n_source_skills.append(subtraj.pop("n_source_skills"))
			n_target_skills.append(subtraj.pop("n_target_skills"))

		subtraj_dict = tree_map(lambda *args: np.stack(args, axis=0), *subtrajectories)
		subtraj_dict["skills_order"] = subtraj_dict["skills_order"].astype("i4")

		buffer_sample = ComDeBufferSample(
			**subtraj_dict,
			source_skills_idxs=np.array(source_skills),  # This is index. Not dense vector
			target_skills_idxs=np.array(target_skills),  # This is index. Not dense vector
			n_source_skills=np.array(n_source_skills),
			n_target_skills=np.array(n_target_skills),
			sequential_requirement=np.array(sequential_requirements),
			non_functionality=np.array(non_functionalities),
			source_parameters=source_parameters,
			parameters=parameters
		)
		return buffer_sample

	@staticmethod
	def get_skills_parameter(
		skills_idxs: np.ndarray,
		parameter: Any,
	) -> np.ndarray:
		"""
		:param skills_idxs:
		:param parameter:
		:return: [seq_len, param_dim]
		"""
		return_parameter = np.zeros_like(skills_idxs)
		for skill_idx, param in parameter.items():
			return_parameter = np.where(skills_idxs == skill_idx, param, return_parameter)
		return_parameter = return_parameter[..., np.newaxis]
		return return_parameter
