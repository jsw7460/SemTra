import random
from typing import Dict

import numpy as np

from comde.comde_modules.base import ComdeBaseModule
from comde.rl.buffers import ComdeBuffer
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv
from comde.trainer.base import BaseTrainer
from comde.trainer.comde_trainer import ComdeTrainer


class BaselineTrainer(BaseTrainer):
	def __init__(
		self,
		cfg: Dict,
		env: SkillInfoEnv,
		baseline: ComdeBaseModule,  # "skill decoder" == "low policy"
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
		self.skill_infos = env.skill_infos
		super(BaselineTrainer, self).__init__(cfg=cfg, env=env)
		self.video_parsing = baseline.video_parsing
		self.baseline = baseline
		self.info_records = {"info/suffix": self.cfg["save_suffix"]}
		self.train_online_context = cfg["train_online_context"]
		if self.train_online_context:
			self.wind_axis = cfg["env"]["wind_axis"]
			# self.winds = [-.6, -.3, 0, .3, .6]
			self.winds = [-0.3, -0.15, 0.0, 0.15, 0.3]
			print("@"*999, self.winds)

	def prepare_run(self):
		super(BaselineTrainer, self).prepare_run()
		skills = [random.choice(sk) for sk in list(self.skill_infos.values())]
		skills.sort(key=lambda sk: sk.index)
		skills = [sk.vec for sk in skills]
		ComdeTrainer.append_dummy_skill(skills)
		self.__last_onehot_skills = np.array([sk for sk in skills])

	def _preprocess_replay_data(self, replay_data: ComDeBufferSample) -> ComDeBufferSample:
		replay_data = ComdeTrainer.get_skill_from_idxs(replay_data, self.__last_onehot_skills)
		replay_data = ComdeTrainer.get_language_guidance_from_template(
			env=self.env,
			replay_data=replay_data,
			video_parsing=self.video_parsing
		)
		return replay_data

	def run(self, replay_buffer: ComdeBuffer):
		for _ in range(self.step_per_dataset):
			replay_data = replay_buffer.sample(self.batch_size)  # type: ComDeBufferSample

			if self.train_online_context:
				winds = np.array(random.choices(self.winds, k=self.batch_size))
				nonstationary_actions = replay_data.actions
				nonstationary_actions[..., self.wind_axis] += winds[..., np.newaxis]
				replay_data = replay_data._replace(actions=nonstationary_actions)

			replay_data = self._preprocess_replay_data(replay_data)
			info = self.baseline.update(replay_data)

			self.record_from_dicts(info, mode="train")
			self.n_update += 1

			if (self.n_update % self.log_interval) == 0:
				self.dump_logs(step=self.n_update)

			if (self.n_update % self.save_interval) == 0:
				self.save()

	def evaluate(self, replay_buffer: ComdeBuffer):
		""" ... """

	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(BaselineTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			cur_step = str(self.n_update)
			getattr(self, key).save(f"{save_path}_{cur_step}")

	def load(self, *args, **kwargs):
		raise NotImplementedError()
