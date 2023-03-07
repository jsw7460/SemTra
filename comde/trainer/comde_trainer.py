from typing import Dict, Union

import numpy as np

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.buffers import ComdeBuffer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.trainer.base import BaseTrainer


class ComdeTrainer(BaseTrainer):
	def __init__(
		self,
		cfg: Dict,
		low_policy: BaseLowPolicy,  # "skill decoder" == "low policy"
		seq2seq: BaseSeqToSeq,
		termination: BaseTermination,
		idx_to_skill: Dict[str, np.ndarray]
	):
		"""
		This trainer consist of three modules
			1. low policy == skill_decoder
			2. skill_seqtoseq
			3. idx to skill
		Modules are defined outside the trainer.
		idx to skill:
		This class is not responsible for fulfilling replay buffer.
		"""
		super(ComdeTrainer, self).__init__(cfg)
		self.low_policy = low_policy
		self.seq2seq = seq2seq
		self.termination = termination
		self.idx_to_skill = idx_to_skill

		self._np_idx_to_skill = np.array(list(self.idx_to_skill.values()))
		self.replay_buffer: Union[ComdeBuffer] = None

		self.info_records = {
			"info/suffix": self.cfg["save_suffix"]
		}

	@property
	def np_idx_to_skill(self):
		return self._np_idx_to_skill

	@np_idx_to_skill.setter
	def np_idx_to_skill(self, *args, **kwargs):
		raise NotImplementedError("This is fixed")

	def update_skill(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		skills_idxs = replay_data.skills_idxs
		source_skills = replay_data.source_skills
		target_skills = replay_data.target_skills

		replay_data = replay_data._replace(
			skills=self.np_idx_to_skill[skills_idxs],
			source_skills=self.np_idx_to_skill[source_skills],
			target_skills=self.np_idx_to_skill[target_skills]
		)

		return replay_data

	def run(self):
		for _ in range(self.step_per_dataset):
			replay_data = self.replay_buffer.sample(self.batch_size)  # type: ComDeBufferSample
			replay_data = self.update_skill(replay_data)
			info1 = self.low_policy.update(replay_data)
			info2 = self.seq2seq.update(replay_data, low_policy=self.low_policy.model)
			info3 = self.termination.update(replay_data)

			self.record_from_dicts(info1, info2, info3, mode="train")

			self.n_update += 1

			if (self.n_update % self.log_interval) == 0:
				self.dump_logs(step=self.n_update)

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(ComdeTrainer, self).dump_logs(step=step)

	def save(self, *args, **kwargs):
		raise NotImplementedError()

	def load(self, *args, **kwargs):
		raise NotImplementedError()
