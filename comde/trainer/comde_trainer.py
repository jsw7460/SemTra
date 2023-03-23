from typing import Dict

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
		skill_to_vec: Dict[str, np.ndarray]
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
		self.idx_to_skill = skill_to_vec

		np_idx_to_skill = np.array(list(self.idx_to_skill.values()))

		if "-1" not in self.idx_to_skill.keys():
			zero_skill = np.zeros_like(list(self.idx_to_skill.values())[0])
			np_idx_to_skill = np.concatenate((np_idx_to_skill, zero_skill[np.newaxis, ...]), axis=0)

		self._np_idx_to_skill = np_idx_to_skill

		self.info_records = {
			"info/suffix": self.cfg["save_suffix"]
		}

	@property
	def np_idx_to_skill(self):
		return self._np_idx_to_skill

	@np_idx_to_skill.setter
	def np_idx_to_skill(self, *args, **kwargs):
		raise NotImplementedError("This is fixed")

	def get_skill_from_idxs(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		skills_idxs = replay_data.skills_idxs
		source_skills = replay_data.source_skills
		target_skills = replay_data.target_skills

		replay_data = replay_data._replace(
			skills=self.np_idx_to_skill[skills_idxs],
			source_skills=self.np_idx_to_skill[source_skills],
			target_skills=self.np_idx_to_skill[target_skills]
		)
		return replay_data

	def run(self, replay_buffer: ComdeBuffer):
		for _ in range(self.step_per_dataset):
			replay_data = replay_buffer.sample(self.batch_size)  # type: ComDeBufferSample
			replay_data = self.get_skill_from_idxs(replay_data)

			# NOTE: Do not change the training order of modules.
			info1 = self.seq2seq.update(replay_data=replay_data, low_policy=self.low_policy.model)
			info2 = self.low_policy.update(replay_data._replace(skills=np.array(info1.pop("__pred_target_skills"))))
			info3 = self.termination.update(replay_data)

			self.record_from_dicts(info1, info2, info3, mode="train")
			self.n_update += 1

			if (self.n_update % self.log_interval) == 0:
				self.dump_logs(step=self.n_update)

	def evaluate(self, replay_buffer: ComdeBuffer):
		eval_data = replay_buffer.sample(128)  # type: ComDeBufferSample

		eval_data = self.get_skill_from_idxs(eval_data)

		info1 = self.seq2seq.evaluate(replay_data=eval_data, np_idx_to_skills=self.np_idx_to_skill.copy())
		seq2seq_output = info1["__seq2seq_output"]
		pred_target_skills = np.take_along_axis(seq2seq_output, eval_data.skills_order[..., np.newaxis], axis=1)

		info2 = self.low_policy.evaluate(
			eval_data._replace(skills=pred_target_skills)
		)
		info3 = self.termination.evaluate(eval_data)

		self.record_from_dicts(info1, info2, info3, mode="eval")
		self.dump_logs(step=self.n_update)

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(ComdeTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			getattr(self, key).save(save_path)

	# self.low_policy.save()

	def load(self, *args, **kwargs):
		raise NotImplementedError()
