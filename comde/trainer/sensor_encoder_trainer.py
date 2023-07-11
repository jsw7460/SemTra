import random
import math
from typing import Dict, List

import numpy as np

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers import ComdeBuffer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv
from comde.trainer.base import BaseTrainer
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation


class SensorEncoderTrainer(BaseTrainer):
	def __init__(
		self,
		cfg: Dict,
		env: SkillInfoEnv,
		sensor_encoder: BaseLowPolicy,  # "skill decoder" == "low policy"
		*args, **kwargs
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
		self.skill_infos = env.skill_infos  # type: Dict[str, List[SkillRepresentation]]
		super(SensorEncoderTrainer, self).__init__(cfg=cfg, env=env)

		self.sensor_encoder = sensor_encoder
		self.info_records = {"info/suffix": self.cfg["save_suffix"]}
		self.idx_to_parameter_dict = env.get_idx_to_parameter_dict()

	def prepare_run(self):
		super(SensorEncoderTrainer, self).prepare_run()

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

	@staticmethod
	def get_language_guidance_from_template(
		env: SkillInfoEnv,
		replay_data: ComDeBufferSample,
		video_parsing: bool = True
	) -> ComDeBufferSample:
		language_guidances = []
		seq_reqs = replay_data.str_sequential_requirement
		nfs = replay_data.str_non_functionality
		params = replay_data.parameters
		sk_idxs = replay_data.source_skills_idxs
		n_skills = replay_data.n_source_skills

		for seq_req, nf, prm, sources, n_sk in zip(seq_reqs, nfs, params, sk_idxs, n_skills):
			language_guidance = env.get_language_guidance_from_template(
				sequential_requirement=seq_req,
				non_functionality=nf,
				parameter={k: v for k, v in prm.items() if k != -1},  # Remove dummy skill
				source_skills_idx=sources[:n_sk],
				video_parsing=video_parsing
			)
			language_guidances.append(language_guidance)

		replay_data = replay_data._replace(language_guidance=language_guidances)
		return replay_data

	def _preprocess_replay_data(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		# Make first few parameter idxs
		skills_idxs = replay_data.initial_few_skills_idxs
		params = replay_data.parameters
		initial_few_parameters_idxs = []

		for skill_idxs, param in zip(skills_idxs, params):
			parameters_idxs = []
			for skill_idx in skill_idxs:
				skill_parameter_dict = self.idx_to_parameter_dict[skill_idx]
				current_skill_param = float(param[skill_idx])
				for k, v in skill_parameter_dict.items():
					if math.fabs(v - current_skill_param) < 1e-9:
						parameters_idxs.append(k)
						break
			initial_few_parameters_idxs.append(np.array(parameters_idxs))

		initial_few_parameters_idxs = np.array(initial_few_parameters_idxs)
		replay_data = replay_data._replace(initial_few_parameters_idxs=initial_few_parameters_idxs)
		return replay_data

	def run(self, replay_buffer: ComdeBuffer):
		for _ in range(self.step_per_dataset):
			replay_data = replay_buffer.sample(self.batch_size)  # type: ComDeBufferSample
			replay_data = self._preprocess_replay_data(replay_data)
			info = self.sensor_encoder.update(replay_data)
			self.record_from_dicts(info, mode="train")
			self.n_update += 1

			if (self.n_update % self.log_interval) == 0:
				self.dump_logs(step=self.n_update)

			if (self.n_update % self.save_interval) == 0:
				self.save()

	def evaluate(self, replay_buffer: ComdeBuffer):
		eval_data = replay_buffer.sample(128)  # type: ComDeBufferSample
		eval_data = self._preprocess_replay_data(eval_data)
		self.dump_logs(step=self.n_update)

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(SensorEncoderTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			cur_step = str(self.n_update)
			getattr(self, key).save(f"{save_path}_{cur_step}")

	def load(self, *args, **kwargs):
		raise NotImplementedError()
