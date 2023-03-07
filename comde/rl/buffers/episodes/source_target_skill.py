from typing import List, Dict

from comde.rl.buffers.episodes.skill import SkillContainedEpisode

import numpy as np


class SourceTargetSkillContainedEpisode(SkillContainedEpisode):
	def __init__(self):
		super(SourceTargetSkillContainedEpisode, self).__init__()

		self.source_skills = []
		self.n_source_skills = 0

		self.target_skills = []
		self.n_target_skills = 0

		self.language_operator = None  # E.g., 'do sequentially', ...

		self.skills_orders = []		# Increasing sequence.

	def __getitem__(self, idx):
		episode = super(SourceTargetSkillContainedEpisode, self).__getitem__(idx)
		if idx.start < 0:
			idx = slice(0, idx.stop, None)
		skills_orders = list(self.skills_orders[idx].copy())
		return SourceTargetSkillContainedEpisode.from_list(
			observations=episode.observations,
			next_observations=episode.next_observations,
			actions=episode.actions,
			rewards=episode.rewards,
			dones=episode.dones,
			infos=episode.infos,
			source_skills=self.source_skills,
			target_skills=self.target_skills,
			language_operator=self.language_operator,
			first_observations=episode.first_observations,
			skills=episode.skills,
			skills_done=episode.skills_done,
			skills_idxs=episode.skills_idxs,
			skills_orders=skills_orders,
			rtgs=episode.rtgs,
			maskings=episode.maskings,
			timesteps=episode.timesteps
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
		language_operator: List = None,
		first_observations: List = None,
		skills: List = None,
		skills_done: List = None,
		skills_idxs: List = None,
		skills_orders: List = None,
		rtgs: List = None,
		maskings: List = None,
		timesteps: List = None,
	) -> "SourceTargetSkillContainedEpisode":
		ret = SourceTargetSkillContainedEpisode()
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
		ret.language_operator = language_operator.copy()

		return ret

	def get_numpy_subtrajectory(self, from_: int, to_: int, batch_mode: bool) -> Dict:
		super_data \
			= super(SourceTargetSkillContainedEpisode, self).get_numpy_subtrajectory(from_, to_, batch_mode=batch_mode)
		current_data = {
			"source_skills": self.source_skills.copy(),
			"language_operator": self.language_operator.copy(),
			"target_skills": self.target_skills.copy(),
			"skills_order": np.array(self.skills_orders[from_: to_]),
			"n_source_skills": self.n_source_skills,
			"n_target_skills": self.n_target_skills
		}
		if batch_mode:
			self.expand_1st_dim(current_data)

		return {**super_data, **current_data}

	def to_numpydict(self) -> Dict:
		ret = super(SourceTargetSkillContainedEpisode, self).to_numpydict()
		ret.update({
			"source_skills": self.source_skills.copy(),
			"language_operator": self.language_operator.copy(),
			"target_skills": self.target_skills.copy()
		})
		return ret

	def set_zeropaddings(self, n_padding: int, max_possible_skills: int = None):
		assert len(self.target_skills) > 0, "We require at least one target skills !"
		super(SourceTargetSkillContainedEpisode, self).set_zeropaddings(n_padding=n_padding)
		zero_skill = np.zeros_like(self.target_skills[0])
		[self.source_skills.append(zero_skill.copy()) for _ in range(max_possible_skills - self.n_source_skills)]
		[self.target_skills.append(zero_skill.copy()) for _ in range(max_possible_skills - self.n_target_skills)]
		[self.skills_orders.append(-1) for _ in range(n_padding)]

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
		timestep: int = None,
		skill_order: np.ndarray = None,
	):
		# "skills" are processed using "skills_idxs" when making minibatch. So we add 'None' skill to buffer.
		skill = np.empty((0,))
		super(SourceTargetSkillContainedEpisode, self).add(
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
			timestep=timestep
		)
		self.skills_orders.append(skill_order.copy())

	def add_from_dict(self, dataset: Dict):
		traj_len = len(dataset["observations"])
		self.source_skills = dataset["source_skills"]
		self.n_source_skills = len(self.source_skills)

		self.target_skills = dataset["target_skills"]
		self.n_target_skills = len(self.target_skills)
		self.language_operator = dataset["language_operator"]

		for i in range(traj_len):
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
				timestep=i,
				skill_order=dataset["skills_order"][i]
			)