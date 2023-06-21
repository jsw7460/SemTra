import random
from typing import Dict, List, Union

from collections import deque

import h5py
import numpy as np

from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.utils import SkillInfoEnv
from comde.trainer.base import BaseTrainer
from comde.trainer.comde_trainer import ComdeTrainer

array = np.array  # DO NOT REMOVE THIS !


class ComposeTrainer(BaseTrainer):

	def __init__(
		self,
		cfg: Dict,
		envs: Dict[str, Union[ComdeSkillEnv, SkillInfoEnv]],
		offset_info: Dict[str, int],
		seq2seq: BaseSeqToSeq
	):
		self.__last_onehot_skills = None
		self.skill_infos = seq2seq.tokens
		super(ComposeTrainer, self).__init__(cfg=cfg, env=envs)
		self.envs = envs
		del self.env
		self.seq2seq = seq2seq
		self.info_records = {"info/suffix": self.cfg["save_suffix"]}

		self.offset_info = offset_info

		self.max_source_skills = cfg["max_source_skills"]
		self.max_target_skills = cfg["max_target_skills"]

		self.buffer = deque(maxlen=100_000)

	def prepare_run(self):
		super(ComposeTrainer, self).prepare_run()

		skills = [sk[0] for sk in list(self.skill_infos.values())]
		skills.sort(key=lambda sk: sk.index)
		skills = [sk.vec for sk in skills]
		ComdeTrainer.append_dummy_skill(skills)

		self.__last_onehot_skills = np.array([sk for sk in skills])

	def _get_env_from_filename(self, filename: str):
		if "metaworld" in filename:
			return self.envs["metaworld"]
		elif "kitchen" in filename:
			return self.envs["kitchen"]
		elif "rlbench" in filename:
			return self.envs["rlbench"]

	def _get_target_skill_vector(self, target_skills_idx: List[int], offset: int) -> List[int]:
		"""
			Because of the offset index, we need to modify it here
		"""
		idxs = []
		for idx in target_skills_idx:
			idxs.append(idx + offset if idx != -1 else -1)
		return idxs

	def make_dataset(self, trajectories: List[str]) -> List[ComDeBufferSample]:
		# source_skills = []
		target_skills = []
		language_guidances = []
		target_skills_idxs = []
		n_source_skills = []
		n_target_skills = []
		for path in trajectories:
			env = self._get_env_from_filename(path)
			trajectory = h5py.File(path, "r")

			sequential_requirement = str(trajectory["sequential_requirement"][()], "utf-8")
			non_functionality = str(trajectory["non_functionality"][()], "utf-8")
			parameter = str(trajectory["parameter"][()], "utf-8")
			parameter = env.eval_param(parameter)

			source_skills_idx = []
			for skills_in_demo in trajectory["source_skills"].values():
				for skill in np.array(skills_in_demo):
					source_skills_idx.append(skill)
			target_skills_idx = list(trajectory["target_skills"])

			n_source_skill = len(source_skills_idx)
			n_target_skill = len(target_skills_idx)

			language_guidance = env.get_language_guidance_from_template(
				sequential_requirement=sequential_requirement,
				non_functionality=non_functionality,
				parameter=parameter,
				source_skills_idx=source_skills_idx
			)

			if language_guidance is None:
				continue

			n_target_pad = self.max_target_skills - len(target_skills_idx)
			target_skills_idx = target_skills_idx + [-1 for _ in range(n_target_pad)]

			n_source_skills.append(n_source_skill)
			n_target_skills.append(n_target_skill)

			offset = self.offset_info[str(env)]
			target_skills_idx = self._get_target_skill_vector(target_skills_idx, offset)
			target_skills_idxs.append(target_skills_idx)
			target_skill = self.__last_onehot_skills[target_skills_idx]
			target_skills.append(target_skill)

			language_guidances.append(language_guidance)

		# Define: language_guidance,target_skills, target_skills_idxs, n_source_skills, n_target_skills
		"""
			Done: language_guidance
		"""
		target_skills = np.array(target_skills)
		target_skills_idxs = np.array(target_skills_idxs)
		n_source_skills = np.array(n_source_skills)
		n_target_skills = np.array(n_target_skills)

		# Make minibatch dataset
		n_chunk = max(len(language_guidances) // self.batch_size, 1)
		_target_skills = np.array_split(target_skills, n_chunk, axis=0)
		_target_skills_idxs = np.array_split(target_skills_idxs, n_chunk, axis=0)
		_n_source_skills = np.array_split(n_source_skills, n_chunk, axis=0)
		_n_target_skills = np.array_split(n_target_skills, n_chunk, axis=0)

		buffer_samples = []
		pos = 0
		for ts, tsi, nss, nts in zip(_target_skills, _target_skills_idxs, _n_source_skills, _n_target_skills):
			language_guidance = language_guidances[pos: pos + len(ts)]
			pos += len(ts)
			buffer_sample = ComDeBufferSample(
				language_guidance=language_guidance,
				target_skills=ts,
				target_skills_idxs=tsi,
				n_source_skills=nss,
				n_target_skills=nts
			)
			buffer_samples.append(buffer_sample)

		return buffer_samples

	def run(self, trajectories: List[str]):
		replay_datas = self.make_dataset(trajectories)
		self.buffer.append(replay_datas[0])

		previous_datas = random.choices(self.buffer, k=len(replay_datas))
		replay_datas = replay_datas + previous_datas
		random.shuffle(replay_datas)

		for _ in range(self.step_per_dataset):
			for replay_data in replay_datas:

				info = self.seq2seq.update(replay_data=replay_data)

				self.record_from_dicts(info, mode="train")
				self.n_update += 1

				if (self.n_update % self.log_interval) == 0:
					self.dump_logs(step=self.n_update)

				if (self.n_update % self.save_interval) == 0:
					self.save()

	def evaluate(self, trajectories: List[str]) -> None:
		replay_datas = self.make_dataset(trajectories)
		replay_data = replay_datas[0]
		eval_info = self.seq2seq.evaluate(replay_data)

		self.record_from_dicts(eval_info, mode="eval")
		self.dump_logs(step=self.n_update)


	def dump_logs(self, step: int):
		self.record(self.info_records)
		super(ComposeTrainer, self).dump_logs(step=step)

	def save(self):
		for key, save_path in self.cfg["save_paths"].items():
			cur_step = str(self.n_update)
			getattr(self, key).save(f"{save_path}_{cur_step}")

	def load(self, *args, **kwargs):
		raise NotImplementedError()
