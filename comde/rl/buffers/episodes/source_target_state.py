from typing import List, Dict

import numpy as np
from comde.rl.buffers.episodes.source_target_skill import SourceTargetSkillContainedEpisode


class SourceStateEpisode(SourceTargetSkillContainedEpisode):
	def __init__(self):
		super(SourceStateEpisode, self).__init__()
		self.source_observations = []
		self.source_actions = []
		self.source_maskings = []

	def __getitem__(self, idx):
		episode = super(SourceStateEpisode, self).__getitem__(idx)
		if idx.start < 0:
			idx = slice(0, idx.stop, None)

		source_observations = list(self.source_observations[idx].copy())
		source_actions = list(self.source_actions[idx].copy())
		source_maskings = list(self.source_maskings[idx].copy())

		SourceStateEpisode.from_list(
			observations=episode.observations,
			next_observations=episode.next_observations,
			actions=episode.actions,
			rewards=episode.rewards,
			dones=episode.dones,
			infos=episode.infos,
			source_skills=episode.source_skills,
			target_skills=episode.target_skills,
			sequential_requirement=episode.sequential_requirement,
			non_functionality=episode.non_functionality,
			source_parameter=self.source_parameter,
			parameter=self.parameter,
			first_observations=episode.first_observations,
			skills=episode.skills,
			skills_done=episode.skills_done,
			skills_idxs=episode.skills_idxs,
			skills_orders=episode.skills_orders,
			rtgs=episode.rtgs,
			maskings=episode.maskings,
			timesteps=episode.timesteps,
			source_observations=source_observations,
			source_actions=source_actions,
			source_maskings=source_maskings,
		)

	@staticmethod
	def from_list(
		observations: List,
		next_observations: List,
		actions: List,
		rewards: List,
		dones: List,
		infos: List,
		source_skills: List = None,
		target_skills: List = None,
		sequential_requirement: List = None,
		non_functionality: List = None,
		source_parameter: Dict = None,
		parameter: Dict = None,
		first_observations: List = None,
		skills: List = None,
		skills_done: List = None,
		skills_idxs: List = None,
		skills_orders: List = None,
		rtgs: List = None,
		maskings: List = None,
		timesteps: List = None,
		source_observations: List = None,
		source_actions: List = None,
		source_maskings: List = None
	) -> "SourceStateEpisode":
		ret = SourceStateEpisode()
		ret.observations = observations.copy()
		ret.next_observations = next_observations.copy()
		ret.actions = actions.copy()
		ret.rewards = rewards.copy()
		ret.dones = dones.copy()
		ret.infos = infos.copy()
		ret.first_observations = first_observations.copy()
		ret.skills = skills.copy()
		ret.skills_done = skills_done.copy()
		ret.skills_idxs = skills_idxs.copy()
		ret.skills_orders = skills_orders.copy()
		ret.rtgs = rtgs.copy()
		ret.maskings = maskings.copy()
		ret.timesteps = timesteps.copy()
		ret.source_skills = source_skills.copy()
		ret.target_skills = target_skills.copy()
		ret.sequential_requirement = sequential_requirement.copy()
		ret.non_functionality = non_functionality.copy()
		ret.source_observations = source_observations.copy()
		ret.source_actions = source_actions.copy()
		ret.source_maskings = source_maskings.copy()
		ret.source_parameter = source_parameter.copy()
		ret.parameter = parameter.copy()
		return ret

	def get_numpy_subtrajectory(self, from_: int, to_: int, batch_mode: bool) -> Dict:
		super_data = super(SourceStateEpisode, self).get_numpy_subtrajectory(from_, to_, batch_mode=batch_mode)

		current_data = {
			"source_observations": np.array(self.source_observations[from_: to_]),
			"source_actions": np.array(self.source_actions[from_: to_]),
			"source_maskings": np.array(self.source_maskings[from_: to_])
		}
		if batch_mode:
			self.expand_1st_dim(current_data)

		return {**super_data, **current_data}

	def set_zeropaddings(self, n_padding: int, max_source_skills: int = None, max_target_skills: int = None):
		super(SourceStateEpisode, self).set_zeropaddings(
			n_padding=n_padding,
			max_source_skills=max_source_skills,
			max_target_skills=max_target_skills
		)
		padding_size = len(self) - len(self.source_observations)
		for _ in range(padding_size):
			self.source_observations.append(np.zeros(self.observation_dim, ))
			self.source_actions.append(np.zeros(self.action_dim, ))
			self.source_maskings.append(np.array(0))

	def add(
		self,
		observation: np.ndarray,
		next_observation: np.ndarray,
		action: np.ndarray,
		reward: np.ndarray,
		done: np.ndarray,
		info: List,
		first_observation: np.ndarray = None,
		skill: np.ndarray = None,
		skill_done: np.ndarray = None,
		skill_idx: float = None,
		param_for_skill: np.ndarray = None,
		timestep: int = None,
		skill_order: np.ndarray = None,
		source_observation: np.ndarray = None,
		source_action: np.ndarray = None
	):
		super(SourceStateEpisode, self).add(
			observation=observation,
			next_observation=next_observation,
			action=action,
			reward=reward,
			done=done,
			info=info,
			first_observation=first_observation,
			skill=skill,
			skill_done=skill_done,
			skill_idx=skill_idx,
			param_for_skill=param_for_skill,
			timestep=timestep,
			skill_order=skill_order
		)

	def add_source(self, source_observation: np.ndarray, source_action: np.ndarray):
		self.source_observations.append(source_observation.copy())
		self.source_actions.append(source_action.copy())
		self.source_maskings.append(np.array(1))

	def add_from_dict(self, dataset: Dict):
		tgt_len = len(dataset["observations"])
		src_len = len(dataset["source_observations"])
		self.source_skills = dataset["source_skills_idxs"]
		self.n_source_skills = len(self.source_skills)

		self.target_skills = dataset["target_skills_idxs"]
		self.n_target_skills = len(self.target_skills)
		self.sequential_requirement = dataset["sequential_requirement"]
		self.non_functionality = dataset["non_functionality"]
		self.parameter = dataset["parameter"].copy()
		self.source_parameter = dataset["source_parameter"].copy()

		for i in range(tgt_len):
			self.add(
				observation=dataset["observations"][i],
				next_observation=dataset["next_observations"][i],
				action=dataset["actions"][i],
				reward=dataset["rewards"][i],
				done=dataset["dones"][i],
				info=dataset["infos"][i],
				first_observation=dataset["first_observations"][i],
				skill_done=dataset["skills_done"][i],
				skill_idx=dataset["skills_idxs"][i],
				param_for_skill=dataset["params_for_skills"][i],
				timestep=i,
				skill_order=dataset["skills_order"][i],
			)

		for i in range(src_len):
			self.add_source(
				source_observation=dataset["source_observations"][i],
				source_action=dataset["source_actions"][i]
			)