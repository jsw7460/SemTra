import pickle
from typing import Dict, List

import numpy as np
from jax.tree_util import tree_map

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.envs.utils import SkillHistoryEnv
from comde.trainer.comde_trainer import ComdeTrainer
from comde.utils.common.lang_representation import SkillRepresentation

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3


class HierarchicalRLAgent(ComdeTrainer):

	def __init__(
		self,
		cfg: Dict,
		env: SkillHistoryEnv,
		low_policy: BaseLowPolicy,
		seq2seq: BaseSeqToSeq,  # Sequence generator
		termination: BaseTermination,
		skill_infos: Dict[str, List[SkillRepresentation]]
	):
		"""
			Trainer for few-shot adaptation with online interaction

			We assume natural language is given after templatized.
			So we have 'sequential requirements' and 'params_for_skills'
		"""

		super(HierarchicalRLAgent, self).__init__(
			cfg=cfg,
			low_policy=low_policy,
			seq2seq=seq2seq,
			termination=termination,
			skill_infos=skill_infos
		)

		rl_cfg = cfg["mode"]
		assert rl_cfg["mode"] == "rl", f"This trainer is for RL, not {rl_cfg['mode']}"

		self.envs = [env]
		self.high_policy = self.seq2seq

		# Source skills
		self._source_skills = [5, 0, 3, 4]
		self._str_source_skills = [env.skill_index_mapping[idx] for idx in self._source_skills]
		self.source_skills = np.empty(0, )  # type: np.ndarray	# [1, M, d]
		self.n_source_skills = np.array([4])

		# Sequential requirements
		self._sequential_requirement = {"main": "sequential", "variation": "Execute these in a particular order."}
		self.sequential_requirement = np.empty(0, )  # type: np.ndarray	# [1, d]

		# Non functionality
		self._non_functionality = "wind"
		self.non_functionality = np.empty(0, )  # type: np.ndarray # [1, d]

		# Parameter
		self.param_dict = {k: 0.0 for k in range(7)}
		param_array = np.array(list(self.param_dict.values()))
		param_array = param_array.reshape(-1, 1)
		self.param_array = param_array

		# Misc
		self.param_repeats = self.low_policy.cfg["param_repeats"]
		self.semantic_skill_dim = self.termination.skill_dim
		self.termination_pred_interval = cfg["mode"]["termination_pred_interval"]
		self.subseq_len = self.low_policy.cfg.get("subseq_len", 1)
		self.use_optimal_next_skill = cfg["mode"]["use_optimal_next_skill"]

		self.build_model()

	def build_model(self):
		self._load_sequential_requirement()
		self._load_source_skills()
		self._load_non_functionality()

	def _load_sequential_requirement(self):
		seq_path = self.cfg["sequential_requirements_path"]

		with open(seq_path, "rb") as f:
			seq_dict = pickle.load(f)

		main = self._sequential_requirement["main"]
		variation = self._sequential_requirement["variation"]

		sequential_requirement = seq_dict[main][variation]
		self.sequential_requirement = sequential_requirement[np.newaxis, ...]

	def _load_source_skills(self):
		skill_path = self.cfg["skill_infos_path"]
		with open(skill_path, "rb") as f:
			skill_dict = pickle.load(f)
		np_skills = [skill_dict[skill][0].vec for skill in self._str_source_skills]
		np_skills = np.array(np_skills)
		np_skills = np_skills[np.newaxis, ...]
		self.source_skills = np_skills

	def _load_non_functionality(self):
		nonfunc_path = self.cfg["non_functionalities_path"]
		with open(nonfunc_path, "rb") as f:
			nonfunc_dict = pickle.load(f)

		# Use first variation
		non_functionality = list(nonfunc_dict[self._non_functionality].values())[0]
		non_functionality = non_functionality[np.newaxis, ...]
		self.non_functionality = non_functionality

	def get_param_for_skills(self, idxs: np.ndarray):
		params = []
		for idx in idxs:
			param = self.param_array[idx]  # [batch, param_dim]
			params.append(param)

		params = np.stack(params, axis=1)  # [batch, n_skills, param_dim]
		params = np.repeat(params, repeats=self.param_repeats, axis=-1)
		return params

	def high_policy_act(self):
		pred_target_skills = self.seq2seq.predict(
			source_skills=self.source_skills,
			sequential_requirement=self.sequential_requirement,
			n_source_skills=self.n_source_skills,
			stochastic=True
		)
		return pred_target_skills

	def get_semantic_skills_from_idxs(self, idxs_list: List[np.ndarray]):
		semantic_skills = []
		# idxs: [batch_size,]
		for idxs in idxs_list:
			str_skills = [env.skill_index_mapping[idx] for env, idx in zip(self.envs, idxs)]
			print("STR SKILLS", str_skills)
			semantic_skill = np.array([self.skill_infos[skill][0].vec for skill in str_skills])
			semantic_skills.append(semantic_skill)
		semantic_skills = np.stack(semantic_skills, axis=1)
		return semantic_skills	# [batch, n_skills, semantic_skill_dim]

	def run(self, *args, **kwargs):
		pred_target_skills = self.high_policy_act()
		idxs = pred_target_skills["pred_nearest_idxs"]	# List. each element (np array) has shape [batch_size, ]

		# === Debug
		env = self.envs[0]
		idxs = [np.array([env.onehot_skills_mapping[sk]]) for sk in env.skill_list]
		# === Debug
		semantic_skills = self.get_semantic_skills_from_idxs(idxs)	# [batch, n_skills, semantic_skill_dim]

		params = self.get_param_for_skills(idxs)  # [batch, M, param_dim]
		non_functionality = np.broadcast_to(self.non_functionality, shape=semantic_skills.shape)

		parameterized_skills = np.concatenate((semantic_skills, non_functionality, params), axis=-1)

		for _ in range(10):
			self.run_episode(parameterized_skills)

	def run_episode(self, target_skills: np.ndarray):
		timestep = 0
		n_envs = target_skills.shape[0]
		cur_skill_pos = np.array([0 for _ in range(n_envs)])

		obs_list = [self.envs[i].reset(target_skills[i][cur_skill_pos[i]]) for i in range(n_envs)]
		obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
		first_observations = obs["observations"][:, -1, ...].copy()

		max_skills = np.array([3 for skills in target_skills])

		done = [False for _ in range(n_envs)]
		rew = np.array([0 for _ in range(n_envs)])
		returns = np.array([0.0 for _ in range(n_envs)])

		while not all(done):
			history_observations = obs["observations"]  # [8, 4, 140]
			history_actions = obs["actions"]  # [8, 4, 4]
			# history_rewards = obs["rewards"]  # [8, 4]
			history_skills = obs["skills"]  # [8, 4, 512]
			history_maskings = obs["maskings"]
			timestep += 1

			done_prev = done.copy()

			if self.use_optimal_next_skill:
				cur_skill_pos = np.min([cur_skill_pos + rew, max_skills], axis=0)

			else:
				if ((timestep - 1) % self.termination_pred_interval) == 0 and (timestep > 30):
					maybe_skill_done = self.termination.predict(
						observations=history_observations[:, -1, ...],  # Current observations
						first_observations=first_observations,
						skills=history_skills[:, -1, :self.semantic_skill_dim],
						binary=True
					)
					skill_done = np.where(maybe_skill_done == 1)[0]
					first_observations[skill_done] = history_observations[skill_done, -1, ...]
					cur_skill_pos = np.min([cur_skill_pos + maybe_skill_done, max_skills], axis=0)

			cur_skill_pos = cur_skill_pos.astype("i4")
			# print("Cur skill pos", cur_skill_pos)
			cur_skills = target_skills[np.arange(n_envs), cur_skill_pos, ...]

			timesteps = np.arange(timestep - self.subseq_len, timestep)[np.newaxis, ...]
			timesteps = np.repeat(timesteps, axis=0, repeats=n_envs)
			timesteps[timesteps < 0] = -1

			actions = self.low_policy.predict(
				observations=history_observations,
				actions=history_actions,
				skills=history_skills,
				maskings=history_maskings,
				timesteps=timesteps,
				to_np=True
			)

			step_results = [env.step(act.copy(), cur_skills[i].copy()) for env, act, i in
							zip(self.envs, actions, range(n_envs))]
			obs_list = [result[I_OBS] for result in step_results]

			obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
			rew = np.stack([result[I_REWARD] for result in step_results])
			done = np.stack([result[I_DONE] for result in step_results])

			done = np.logical_or(done, done_prev).astype(np.int32)
			rew_mul_done = np.logical_and(done, done_prev).astype(np.int32)
			rew = rew * (1 - rew_mul_done)
			returns += rew

		print("Returns", returns)
		print("\n\n\n")


	@classmethod
	def load_pretrained_modules(cls, cfg: Dict):
		import pickle
		from pathlib import Path
		from hydra.utils import get_class

		_env_name = cfg["env"]["name"]
		_pretrained_model = cfg["mode"]["pretrained_model"][_env_name]
		date = _pretrained_model["date"]
		suffix = _pretrained_model["suffix"]
		step = _pretrained_model["step"]

		basename = f"{suffix}_{step}"
		cfg_path = Path(cfg["save_prefix"]) / Path(date) / Path("cfg") / Path(f"cfg_{suffix}")

		with open(cfg_path, "rb") as f:
			pretrained_cfg = pickle.load(f)

		pretrained_modules = cfg["mode"]["pretrained_modules"]
		pretrained_dict = dict()

		for module_name in pretrained_modules:
			path = Path(cfg["save_prefix"]) / Path(date) / Path(module_name) / Path(basename)
			module = get_class(pretrained_cfg[module_name]["_target_"])  # type: Union[type, Type[IJaxSavable]]
			module = module.load(path)
			pretrained_dict[module_name] = module

		return pretrained_dict