from typing import List, Dict

import numpy as np

from comde.rl.buffers.episodes.base import Episode


class SkillContainedEpisode(Episode):
	def __init__(self):
		super(SkillContainedEpisode, self).__init__()
		self.first_observations = []
		self.skills = []
		self.skills_done = []
		self.skills_idxs = []
		self.rtgs = []
		self.timesteps = []
		self.representative = None

	def __getitem__(self, idx):
		episode = super(SkillContainedEpisode, self).__getitem__(idx)
		if idx.start < 0:
			idx = slice(0, idx.stop, None)
		first_observations = list(self.first_observations[idx].copy())
		skills = list(self.skills[idx].copy())
		skills_done = list(self.skills_done[idx].copy())
		skills_idxs = list(self.skills_idxs[idx].copy())
		rtgs = list(self.rtgs[idx].copy())
		maskings = list(self.maskings[idx].copy())
		timesteps = list(self.timesteps[idx].copy())

		return SkillContainedEpisode.from_list(
			observations=episode.observations,
			next_observations=episode.next_observations,
			actions=episode.actions,
			rewards=episode.rewards,
			dones=episode.dones,
			infos=episode.infos,
			first_observations=first_observations,
			skills=skills,
			skills_done=skills_done,
			skills_idxs=skills_idxs,
			rtgs=rtgs,
			maskings=maskings,
			timesteps=timesteps
		)

	def get_numpy_subtrajectory(self, from_: int, to_: int, batch_mode: bool) -> Dict:
		super_data = super(SkillContainedEpisode, self).get_numpy_subtrajectory(from_, to_, batch_mode=batch_mode)
		current_data = {
			"first_observations": np.array(self.first_observations[from_: to_]),
			"skills": np.array(self.skills[from_: to_]),
			"skills_done": np.array(self.skills_done[from_: to_]),
			"skills_idxs": np.array(self.skills_idxs[from_: to_]),
			"rtgs": np.array(self.rtgs[from_: to_]),
			"timesteps": np.array(self.timesteps[from_: to_]),
		}
		if batch_mode:
			for key, array in current_data.items():
				current_data.update({key: array[np.newaxis, ...]})

		data = {**super_data, **current_data}
		return data

	def is_from_same_mdp(self, episode: "SkillContainedEpisode") -> bool:
		if self.representative is None:
			return False
		else:
			return self.representative == episode.representative

	@property
	def skill_dim(self):
		return self.skills[0].shape[-1]

	@staticmethod
	def get_rtgs_by_n_skills(skills_idxs: List) -> np.ndarray:
		rtgs = np.zeros((len(skills_idxs),))
		for i in range(len(skills_idxs)):
			rtgs[i] = len(set(skills_idxs[i:]))
		return rtgs

	@staticmethod
	def from_list(
		observations: List,
		next_observations: List,
		actions: List,
		rewards: List,
		dones: List,
		infos: List,
		first_observations: List = None,
		skills: List = None,
		skills_done: List = None,
		skills_idxs: List = None,
		rtgs: List = None,
		maskings: List = None,
		timesteps: List = None
	) -> "SkillContainedEpisode":
		# assert len(observations) \
		# 	   == len(next_observations) \
		# 	   == len(actions) \
		# 	   == len(rewards) \
		# 	   == len(dones) \
		# 	   == len(infos) \
		# 	   == len(first_observations) \
		# 	   == len(skills) \
		# 	   == len(skills_done) \
		# 	   == len(skills_idxs) \
		# 	   == len(rtgs)
		ret = SkillContainedEpisode()
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
		ret.rtgs = rtgs.copy()
		ret.maskings = maskings.copy()
		ret.timesteps = timesteps.copy()
		return ret

	def set_rtgs_by_n_skills(self):
		self.rtgs = self.get_rtgs_by_n_skills(self.skills_idxs).tolist()

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
	):
		super(SkillContainedEpisode, self).add(
			observation=observation,
			next_observation=next_observation,
			action=action,
			reward=reward,
			done=done,
			info=info
		)

		self.first_observations.append(first_observation.copy())
		self.skills.append(skill.copy())
		self.skills_done.append(skill_done.copy())
		self.skills_idxs.append(skill_idx)
		self.timesteps.append(np.array(timestep))

	def to_numpydict(self) -> Dict:
		data = {
			"observations": np.array(self.observations.copy()),
			"next_observations": np.array(self.actions.copy()),
			"actions": np.array(self.actions.copy()),
			"rewards": np.array(self.rewards.copy()),
			"dones": np.array(self.dones.copy()),
			"infos": self.infos.copy(),
			"first_observations": np.array(self.first_observations.copy()),
			"skills": np.array(self.skills.copy()),
			"skills_done": np.array(self.skills_done.copy()),
			"skills_idxs": np.array(self.skills_idxs.copy()),
			"rtgs": np.array(self.rtgs.copy()),
			"maskings": np.array(self.maskings.copy()),
			"timesteps": np.array(self.timesteps.copy())
		}
		return data

	def set_zeropaddings(self, n_padding: int):
		super(SkillContainedEpisode, self).set_zeropaddings(n_padding=n_padding)
		for i in range(n_padding):
			self.first_observations.append(np.zeros(self.observation_dim, ))
			self.skills.append(np.zeros(self.skill_dim, ))
			self.skills_done.append(np.array(False))
			self.skills_idxs.append(np.array(-1))
			self.rtgs.append(np.array(0))
			self.timesteps.append(np.array(-1))
