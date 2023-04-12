import random
from typing import Dict, List

import numpy as np

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.buffers import ComdeBuffer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.trainer.base import BaseTrainer
from comde.utils.common.lang_representation import SkillRepresentation


class ComdeTrainer(BaseTrainer):
	def __init__(
		self,
		cfg: Dict,
		low_policy: BaseLowPolicy,  # "skill decoder" == "low policy"
		seq2seq: BaseSeqToSeq,
		termination: BaseTermination,
		skill_infos: Dict[str, List[SkillRepresentation]]
	):
		"""
		This trainer consist of three modules
			1. low policy == skill_decoder
			2. skill_seqtoseq
			3. termination
		Modules are defined outside the trainer.
		idx to skill: A dictionary, index to (clip) skill
		This class is not responsible for fulfilling replay buffer.
		"""
		super(ComdeTrainer, self).__init__(cfg)
		self.low_policy = low_policy
		self.seq2seq = seq2seq
		self.termination = termination
		self.skill_infos = skill_infos  # type: Dict[str, List[SkillRepresentation]]

		self.info_records = {"info/suffix": self.cfg["save_suffix"]}

		self.__last_onehot_skills = None

	def get_skill_from_idxs(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:

		if self.__last_onehot_skills is None:
			skills = [random.choice(sk) for sk in list(self.skill_infos.values())]
			skills.sort(key=lambda sk: sk.index)
			self.__last_onehot_skills = np.array([sk.vec for sk in skills])

		skills_idxs = replay_data.skills_idxs
		source_skills_idxs = replay_data.source_skills_idxs
		target_skills_idxs = replay_data.target_skills_idxs

		replay_data = replay_data._replace(
			skills=self.__last_onehot_skills[skills_idxs],
			source_skills=self.__last_onehot_skills[source_skills_idxs],
			target_skills=self.__last_onehot_skills[target_skills_idxs]
		)

		return replay_data

	def run(self, replay_buffer: ComdeBuffer):
		for _ in range(self.step_per_dataset):
			replay_data = replay_buffer.sample(self.batch_size)  # type: ComDeBufferSample
			replay_data = self.get_skill_from_idxs(replay_data)

			# NOTE: Do not change the training order of modules.
			info = self.seq2seq.update(replay_data=replay_data, low_policy=self.low_policy)
			# info.update(self.intent_emb.update(replay_data=replay_data, low_policy=self.low_policy.model))
			replay_data = replay_data._replace(intents=info.pop("__intent_for_skill"))
			info.update(self.low_policy.update(replay_data))
			info.update(self.termination.update(replay_data))

			self.record_from_dicts(info, mode="train")
			self.n_update += 1

			if (self.n_update % self.log_interval) == 0:
				self.dump_logs(step=self.n_update)

	def evaluate(self, replay_buffer: ComdeBuffer):
		eval_data = replay_buffer.sample(128)  # type: ComDeBufferSample

		eval_data = self.get_skill_from_idxs(eval_data)

		info1 = self.seq2seq.evaluate(replay_data=eval_data)
		intents = info1["__intents"]

		info2 = self.low_policy.evaluate(replay_data=eval_data._replace(intents=intents))
		info3 = self.termination.evaluate(replay_data=eval_data)

		self.record_from_dicts(info1, info2, info3, mode="eval")
		self.dump_logs(step=self.n_update)

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(ComdeTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			getattr(self, key).save(save_path)

	def load(self, *args, **kwargs):
		raise NotImplementedError()
