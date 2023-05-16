from collections import defaultdict
from typing import List

import numpy as np

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.envs.utils import TimeLimitEnv

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3


def evaluate_flatbc(
	envs: List[TimeLimitEnv],
	baseline: BaseLowPolicy,
	semantic_skills_sequence: np.ndarray,  # [n_envs, n_target_skills, d]
	param_for_skill: np.ndarray,  # [n_envs, n_target_skills, d]
	save_results: bool = False
):
	envs = [env.env for env in envs]	# unwrap history env
	# Some variables

	n_envs = len(envs)

	cur_skill_pos = np.array([0 for _ in range(n_envs)])  # [8, ]
	target_parameterized_skills = np.concatenate((semantic_skills_sequence, param_for_skill), axis=-1)
	n_possible_skills = target_parameterized_skills.shape[1]
	max_skills = np.array([n_possible_skills - 1 for _ in range(n_envs)])

	# Prepare save
	eval_infos = {f"env_{k}": defaultdict(list, env_name=envs[k].get_short_str_for_save()) for k in range(n_envs)}

	# Prepare env
	timestep = 0
	done = [False for _ in range(n_envs)]
	rew = np.array([0 for _ in range(n_envs)])

	observations = np.stack([envs[i].reset() for i in range(n_envs)])

	returns = np.array([0.0 for _ in range(n_envs)])

	while not all(done):

		timestep += 1
		done_prev = done.copy()
		cur_skill_pos = np.min([cur_skill_pos + rew, max_skills], axis=0)
		actions = baseline.predict(
			observations=observations,
			target_parameterized_skills=target_parameterized_skills,
			cur_skill_pos=cur_skill_pos.reshape(-1, 1),
			to_np=True
		)

		step_results = [env.step(act.copy()) for env, act, i in zip(envs, actions, range(n_envs))]
		obs_list = [result[I_OBS] for result in step_results]

		observations = np.stack([result[I_OBS] for result in step_results])
		rew = np.stack([result[I_REWARD] for result in step_results])

		done = np.stack([result[I_DONE] for result in step_results])
		if save_results:
			for k in range(n_envs):
				eval_infos[f"env_{k}"]["observations"].append(obs_list[k])
				eval_infos[f"env_{k}"]["actions"].append(actions[k])
				eval_infos[f"env_{k}"]["rewards"].append(rew[k])
				eval_infos[f"env_{k}"]["infos"].append(step_results[k][I_INFO])
		else:
			for k in range(n_envs):
				eval_infos[f"env_{k}"]["rewards"].append(rew[k])

		# Advance state.
		done = np.logical_or(done, done_prev).astype(np.int32)
		rew_mul_done = np.logical_and(done, done_prev).astype(np.int32)
		rew = rew * (1 - rew_mul_done)
		returns += rew

	n_tasks = sum([env.n_target for env in envs])
	eval_fmt = f"Returns: {returns} \n" \
			   f"Total sum: {returns.sum()} among {len(envs)} tasks \n" \
			   f"Total success ratio: {100 * (returns.sum() / n_tasks)}"
	return eval_infos, eval_fmt
