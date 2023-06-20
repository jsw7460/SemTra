import random
from typing import Dict, List, Optional

import numpy as np

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.buffers import ComdeBuffer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv
from comde.trainer.base import BaseTrainer
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation


class ComdeTrainer(BaseTrainer):
	def __init__(
		self,
		cfg: Dict,
		env: SkillInfoEnv,
		low_policy: BaseLowPolicy,  # "skill decoder" == "low policy"
		termination: BaseTermination,
		seq2seq: Optional[BaseSeqToSeq] = None,
		# skill_infos: Dict[str, List[SkillRepresentation]]
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
		self.__last_onehot_skills = None
		self.skill_infos = env.skill_infos  # type: Dict[str, List[SkillRepresentation]]
		super(ComdeTrainer, self).__init__(cfg=cfg, env=env)

		self.low_policy = low_policy
		self.seq2seq = seq2seq
		self.termination = termination

		self.info_records = {"info/suffix": self.cfg["save_suffix"]}

	def prepare_run(self):
		super(ComdeTrainer, self).prepare_run()
		skills = [random.choice(sk) for sk in list(self.skill_infos.values())]
		skills.sort(key=lambda sk: sk.index)
		skills = [sk.vec for sk in skills]
		self.append_dummy_skill(skills)
		self.__last_onehot_skills = np.array([sk for sk in skills])

	@staticmethod
	def append_dummy_skill(skills: List[np.array]):
		dummy_skill = np.zeros_like(skills[0])
		skills.append(dummy_skill)

	@staticmethod
	def get_skill_from_idxs(
		replay_data: ComDeBufferSample,
		last_onehot_skills: Dict
	) -> ComDeBufferSample:
		# Index -> Vector mapping
		skills_idxs = replay_data.skills_idxs
		source_skills_idxs = replay_data.source_skills_idxs
		target_skills_idxs = replay_data.target_skills_idxs

		replay_data = replay_data._replace(
			skills=last_onehot_skills[skills_idxs],
			source_skills=last_onehot_skills[source_skills_idxs],
			target_skills=last_onehot_skills[target_skills_idxs]
		)
		return replay_data

	def _get_language_guidance_from_template(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		language_guidances = []
		seq_reqs = replay_data.str_sequential_requirement
		nfs = replay_data.str_non_functionality
		params = replay_data.parameters
		sk_idxs = replay_data.source_skills_idxs
		n_skills = replay_data.n_source_skills

		for seq_req, nf, prm, sources, n_sk in zip(seq_reqs, nfs, params, sk_idxs, n_skills):
			language_guidance = self.env.get_language_guidance_from_template(
				sequential_requirement=seq_req,
				non_functionality=nf,
				parameter={k: v for k, v in prm.items() if k != -1},  # Remove dummy skill
				source_skills_idx=sources[:n_sk]
			)
			language_guidances.append(language_guidance)

		replay_data = replay_data._replace(language_guidance=language_guidances)
		return replay_data

	def _preprocess_replay_data(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		replay_data = self.get_skill_from_idxs(replay_data, self.__last_onehot_skills)
		replay_data = self._get_language_guidance_from_template(replay_data)

		return replay_data

	def run(self, replay_buffer: ComdeBuffer):
		for _ in range(self.step_per_dataset):
			replay_data = replay_buffer.sample(self.batch_size)  # type: ComDeBufferSample
			replay_data = self._preprocess_replay_data(replay_data)

			# NOTE: Do not change the training order of modules.
			if self.cfg["update_seq2seq"]:
				info = self.seq2seq.update(replay_data=replay_data, low_policy=self.low_policy)
			else:
				info = dict()
			replay_data = replay_data._replace(parameterized_skills=None)

			info.update(self.low_policy.update(replay_data))
			info.update(self.termination.update(replay_data))

			self.record_from_dicts(info, mode="train")
			self.n_update += 1

			if (self.n_update % self.log_interval) == 0:
				self.dump_logs(step=self.n_update)

			if (self.n_update % self.save_interval) == 0:
				self.save()

	def evaluate(self, replay_buffer: ComdeBuffer):
		eval_data = replay_buffer.sample(128)  # type: ComDeBufferSample
		eval_data = self._preprocess_replay_data(eval_data)

		if self.cfg["update_seq2seq"]:
			info1 = self.seq2seq.evaluate(replay_data=eval_data)
		else:
			info1 = dict()
		info1.update({"__parameterized_skills": None})
		parameterized_skills = info1["__parameterized_skills"]

		info2 = self.low_policy.evaluate(
			replay_data=eval_data._replace(parameterized_skills=parameterized_skills),
		)
		info3 = self.termination.evaluate(replay_data=eval_data)

		self.record_from_dicts(info1, info2, info3, mode="eval")
		self.dump_logs(step=self.n_update)

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(ComdeTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			cur_step = str(self.n_update)
			getattr(self, key).save(f"{save_path}_{cur_step}")

	def load(self, *args, **kwargs):
		raise NotImplementedError()
