from typing import Dict, Union
from comde.trainer.base import BaseTrainer
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.rl.buffers import SkillEpisodicMaskingBuffer, EpisodicMaskingBuffer


class ComdeTrainer(BaseTrainer):
	def __init__(
		self,
		cfg: Dict,
		skill_decoder: BaseLowPolicy,	# "skill decoder" == "low policy"
		skill_seqtoseq: BaseSeqToSeq,
		replay_buffer: Union[SkillEpisodicMaskingBuffer, EpisodicMaskingBuffer]
	):
		"""
		This trainer consist of three modules
			1. skill_decoder
			2. skill_seqtoseq
			3. replay_buffer
		Modules are defined outside the trainer.
		"""
		super(ComdeTrainer, self).__init__(cfg)
		self.skill_decoder = skill_decoder
		self.skill_seqtoseq = skill_seqtoseq
		self.replay_buffer = replay_buffer

	def run(self):
		raise NotImplementedError()

	def save(self, *args, **kwargs):
		raise NotImplementedError()

	def load(self, *args, **kwargs):
		raise NotImplementedError()
