from typing import Dict, Optional, Union

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

	@property
	def np_idx_to_skill(self):
		return self._np_idx_to_skill

	@np_idx_to_skill.setter
	def np_idx_to_skill(self, *args, **kwargs):
		raise NotImplementedError("This is fixed")

	def update_skill(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		skills_idxs = replay_data.skills_idxs
		replay_data._replace(skills=self.np_idx_to_skill[skills_idxs])

		return replay_data

	def run(self):
		replay_data = self.replay_buffer.sample(self.batch_size)	# type: ComDeBufferSample
		replay_data = self.update_skill(replay_data)
		self.low_policy.update(replay_data)
		self.seq2seq.update(replay_data)
		self.termination.update(replay_data)

	def save(self, *args, **kwargs):
		raise NotImplementedError()

	def load(self, *args, **kwargs):
		raise NotImplementedError()
