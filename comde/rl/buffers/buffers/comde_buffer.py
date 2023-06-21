import pickle
from termcolor import colored
import random
from copy import deepcopy
from typing import Dict, Optional, Union, Tuple, List

import h5py
import numpy as np
from jax.tree_util import tree_map
from stable_baselines3.common.vec_env import VecNormalize

from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.rl.buffers.buffers.episodic import EpisodicMaskingBuffer
from comde.rl.buffers.episodes.source_target_skill import SourceTargetSkillContainedEpisode
from comde.rl.buffers.episodes.source_target_state import SourceStateEpisode
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv
from comde.utils.common.misc import get_params_for_skills

array = np.array  # DO NOT REMOVE THIS !


class ComdeBuffer(EpisodicMaskingBuffer):
	MUST_LOADED_COMPONENTS = {
		"actions", "skills_idxs", "skills_order", "skills_done", "source_skills", "target_skills"
	}

	def __init__(
		self,
		env: Union[ComdeSkillEnv, SkillInfoEnv],
		subseq_len: int,
		cfg: Dict,
		n_envs: int = 1,
		buffer_size: int = -1,  # No matter
	):
		super(ComdeBuffer, self).__init__(
			env=env,
			subseq_len=subseq_len,
			n_envs=n_envs,
			buffer_size=buffer_size,
			use_all_previous_components=False
		)
		del self.representative_to_indices

		cfg = deepcopy(cfg)

		self.num_skills_done_relabel = cfg["num_skills_done_relabel"]
		self.max_source_skills = cfg["max_source_skills"]
		self.max_target_skills = cfg["max_target_skills"]
		self.save_source_trajectory = cfg["save_source_trajectory"]
		self.default_source_skills = env.get_default_parameter()
		# self.default_source_skills \
		# 	= cfg["default_source_skills"]  # type: Dict[str, Dict[int, Union[float, np.ndarray]]]

		for nf, pdict in self.default_source_skills.items():
			if -1 in pdict.keys():
				raise LookupError("-1 is for the skill which is padded. Please fix here.")
			else:
				pdict.update({-1: np.zeros_like(np.array(list(pdict.values())[0]))})
		self.observation_keys = cfg["observation_keys"]
		# If parameter is numpy -> use eval function, otherwise ast.literal_eval
		self.eval_param = self.env.eval_param
		self.episode_class = SourceStateEpisode if self.save_source_trajectory else SourceTargetSkillContainedEpisode

	def add_dict_chunk(self, dataset: Dict, representative: str = None, clear_info: bool = False) -> None:
		raise NotImplementedError("This is only for pickle file. ComDe does not support it.")

	def add_episodes_from_h5py(
		self,
		paths: Dict[str, Union[List, str]],
		sequential_requirements_mapping: Dict,
		non_functionalities_mapping: Dict,
		guidance_to_prm: Optional[BaseSeqToSeq] = None,

	):
		"""
			## README ##
			- Each path in paths corresponds to one trajectory. ('done' one time)
			- "skills" are processed using "skills_idxs" when making minibatch. So we add 'None' skill to buffer.
		"""
		trajectory_paths = paths["trajectory"]

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
				guidance_to_prm=guidance_to_prm
			)
			if dataset is None:
				continue
			episode.add_from_dict(dataset)
			episode.set_rtgs()

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
		guidance_to_prm: Optional[BaseSeqToSeq] = None
	) -> Dict:
		must_loaded_components = self.MUST_LOADED_COMPONENTS.union(set(self.observation_keys))
		assert must_loaded_components <= trajectory.keys(), \
			f"Under qualified dataset. Please fill {must_loaded_components - trajectory.keys()}"

		obs_values = []
		for obs_key in self.observation_keys:
			obs_values.append(trajectory[obs_key])

		observations = np.hstack(obs_values)
		next_observations = np.zeros_like(observations)
		next_observations[: -1] = observations[1:]
		actions = np.array(trajectory["actions"])
		actions = self.env.get_buffer_action(actions)

		traj_len = len(observations)

		rtgs = np.zeros((traj_len,))
		dones = np.zeros((traj_len,), dtype=bool)

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
		non_functionality = str(trajectory["non_functionality"][()], "utf-8")	# "speed"
		skills_idxs = np.array(trajectory["skills_idxs"])
		optimal_parameter = str(trajectory["parameter"][()], "utf-8")	#
		optimal_parameter = self.eval_param(optimal_parameter)
		"""
			Note: Here, I have to define 'parameter' variable using the 
			prediction of prompt learning model.
		"""
		parameter = optimal_parameter

		if -1 in parameter.keys():
			raise LookupError("Skill index -1 is for the skill which is padded. Please fix here.")
		else:
			parameter.update({-1: 0.0})

		source_parameter = self.default_source_skills[non_functionality].copy()

		params_for_skills = get_params_for_skills(
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
		skills_done = np.array(trajectory["skills_done"])
		done_times = np.where(np.array(trajectory["skills_done"]) == 1)[0]
		augmented_skills_done = skills_done.copy()

		for timestep in done_times.astype("i4"):
			augmented_skills_done[timestep - num_skills_done_relabel: timestep + num_skills_done_relabel + 1] = 1

		# === Set reward (For some environment this is required)
		if "rewards" in trajectory.keys():
			rewards = np.array(trajectory["rewards"])
		else:
			rewards = skills_done.astype(np.float32)

		dataset = {
			"observations": observations,
			"next_observations": next_observations,
			"actions": actions,
			"rewards": rewards,
			"dones": dones,
			"infos": infos,
			"source_skills_idxs": source_skills,
			"target_skills_idxs": target_skills,
			"language_guidance": None,  # Late binding
			"sequential_requirement": sequential_requirement_vector,  # Sequential requirements act as an operator.
			"str_sequential_requirement": sequential_requirement,
			"non_functionality": non_functionality_vector,
			"str_non_functionality": non_functionality,
			"first_observations": first_observations,
			"skills_done": augmented_skills_done,
			"skills_order": np.array(trajectory["skills_order"]),
			"skills_idxs": skills_idxs,
			"params_for_skills": params_for_skills,
			"rtgs": rtgs,
			"source_parameter": source_parameter,
			"parameter": parameter
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
		language_guidances = []
		str_sequential_requirements = []
		str_non_functionalities = []
		sequential_requirements = []
		non_functionalities = []
		subtrajectories = []
		source_parameters = []
		parameters = []

		for ep, start_idx in zip(episodes, start_idxs):
			subtraj = ep.get_numpy_subtrajectory(from_=start_idx, to_=start_idx + self.subseq_len, batch_mode=False)
			source_parameters.append(subtraj.pop("source_parameter"))
			parameters.append(subtraj.pop("parameter"))
			subtrajectories.append(subtraj)
			source_skills.append(subtraj.pop("source_skills"))
			target_skills.append(subtraj.pop("target_skills"))
			language_guidances.append(subtraj.pop("language_guidance"))
			str_sequential_requirements.append(subtraj.pop("str_sequential_requirement"))
			str_non_functionalities.append(subtraj.pop("str_non_functionality"))
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
			language_guidance=language_guidances,
			str_sequential_requirement=str_sequential_requirements,
			sequential_requirement=np.array(sequential_requirements),
			str_non_functionality=str_non_functionalities,
			non_functionality=np.array(non_functionalities),
			source_parameters=source_parameters,
			parameters=parameters
		)
		return buffer_sample

	# @staticmethod
	# def get_params_for_skills(
	# 	skills_idxs: np.ndarray,  # [l, ]
	# 	parameter: Dict,
	# ) -> np.ndarray:
	# 	"""
	# 	:param skills_idxs:	# [sequence length, ]
	# 	:param parameter:
	# 	:return: [seq_len, param_dim]
	# 	"""
	#
	# 	seq_len = skills_idxs.shape[0]
	# 	raw_param_dim = np.array([list(parameter.values())[0]]).shape[-1]
	# 	return_parameter = np.zeros((seq_len, raw_param_dim))
	# 	for skill_idx, param in parameter.items():
	# 		idxs = np.where(skills_idxs == skill_idx)
	# 		return_parameter[idxs] = param
	#
	# 	return return_parameter
