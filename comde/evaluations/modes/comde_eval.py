from collections import defaultdict
from typing import List

import numpy as np
from jax.tree_util import tree_map

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.environment_encoder.base import BaseEnvEncoder
from comde.comde_modules.termination.base import BaseTermination
from comde.evaluations.utils.postprocess_evaldata import postprocess_eval_data as postprocess
from comde.rl.envs.utils import SkillHistoryEnv

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3


def evaluate_comde(
	envs: List[SkillHistoryEnv],
	low_policy: BaseLowPolicy,
	termination: BaseTermination,
	target_skills: np.ndarray,
	termination_pred_interval: int,
	env_encoder: BaseEnvEncoder = None,
	save_results: bool = False,
	use_optimal_next_skill: bool = False,
):
	# Some variables
	n_envs = len(envs)
	subseq_len = envs[0].num_stack_frames
	semantic_skill_dim = termination.skill_dim

	n_possible_skills = target_skills.shape[1]

	cur_skill_pos = np.array([0 for _ in range(n_envs)])  # [8, ]
	max_skills = np.array([n_possible_skills - 1 for _ in range(n_envs)])

	if env_encoder is not None:
		online_context_dim = low_policy.online_context_diim
	else:
		online_context_dim = 0

	# Prepare save
	eval_infos = {f"env_{k}": defaultdict(list, env_name=envs[k].get_short_str_for_save()) for k in range(n_envs)}

	# Prepare env
	timestep = 0
	done = [False for _ in range(n_envs)]
	rew = np.array([0 for _ in range(n_envs)])

	obs_list = [envs[i].reset(
		np.concatenate((target_skills[i][cur_skill_pos[i]], np.zeros((online_context_dim,))), axis=-1)
	) for i in range(n_envs)]
	obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
	first_observations = obs["observations"][:, -1, ...].copy()
	returns = np.array([0.0 for _ in range(n_envs)])

	while not all(done):
		history_observations = obs["observations"]  # [8, 4, 140]
		history_actions = obs["actions"]  # [8, 4, 4]
		# history_rewards = obs["rewards"]  # [8, 4]
		history_skills = obs["skills"]  # [8, 4, 512]
		history_maskings = obs["maskings"]
		timestep += 1

		# print("History maskings", history_maskings[0])
		# if timestep >= 21:
		# 	exit()
		if env_encoder is not None:
			encoder_input = np.concatenate((history_observations[:, :-1, ...], history_actions[:, :-1, ...]),axis=-1)
			online_context = env_encoder.predict(
				sequence=encoder_input,
				n_iter=np.sum(history_maskings[:, :-1], axis=-1, dtype="i4")
			)
		else:
			online_context = np.zeros(shape=(n_envs, 0))

		done_prev = done.copy()
		if use_optimal_next_skill:
			cur_skill_pos = np.min([cur_skill_pos + rew, max_skills], axis=0)
		else:
			if ((timestep - 1) % termination_pred_interval) == 0 and (timestep > 30):
				maybe_skill_done = termination.predict(
					observations=history_observations[:, -1, ...],  # Current observations
					first_observations=first_observations,
					skills=history_skills[:, -1, :semantic_skill_dim],
					binary=True
				)
				skill_done = np.where(maybe_skill_done == 1)[0]
				first_observations[skill_done] = history_observations[skill_done, -1, ...]
				cur_skill_pos = np.min([cur_skill_pos + maybe_skill_done, max_skills], axis=0)

		cur_skill_pos = cur_skill_pos.astype("i4")
		cur_skills = target_skills[np.arange(n_envs), cur_skill_pos, ...]

		timesteps = np.arange(timestep - subseq_len, timestep)[np.newaxis, ...]
		timesteps = np.repeat(timesteps, axis=0, repeats=n_envs)
		timesteps[timesteps < 0] = -1

		actions = low_policy.predict(
			observations=history_observations,
			actions=history_actions,
			skills=history_skills,
			maskings=history_maskings,
			timesteps=timesteps,
			to_np=True
		)
		step_results = [
			env.step(act.copy(), np.concatenate((cur_skills[i].copy(), online_context[i].copy()), axis=-1))
			for env, act, i in zip(envs, actions, range(n_envs))
		]
		obs_list = [result[I_OBS] for result in step_results]

		obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
		done = np.stack([result[I_DONE] for result in step_results])

		# Advance state.
		done = np.logical_or(done, done_prev).astype(np.int32)
		rew_mul_done = np.logical_and(done, done_prev).astype(np.int32)
		rew = np.stack([result[I_REWARD] for result in step_results])
		rew = rew * (1 - rew_mul_done)
		returns += rew

		if save_results:
			for k in range(n_envs):
				eval_infos[f"env_{k}"]["observations"].append(obs_list[k])
				eval_infos[f"env_{k}"]["actions"].append(actions[k])
				eval_infos[f"env_{k}"]["rewards"].append(rew[k])
				eval_infos[f"env_{k}"]["infos"].append(step_results[k][I_INFO])
				eval_infos[f"env_{k}"]["dones"].append(done[k])
		else:
			for k in range(n_envs):
				eval_infos[f"env_{k}"]["rewards"].append(rew[k])

	n_tasks = sum([env.n_target for env in envs])
	return postprocess(eval_infos=eval_infos, n_tasks=n_tasks)
