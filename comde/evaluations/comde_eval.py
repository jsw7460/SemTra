from collections import defaultdict
from typing import List

import numpy as np
from jax.tree_util import tree_map

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.envs.utils import SkillHistoryEnv

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3


def evaluate_comde_batch(
	envs: List[SkillHistoryEnv],
	low_policy: BaseLowPolicy,
	termination: BaseTermination,
	target_skills: np.ndarray,
	termination_pred_interval: int = 50,
	save_results: bool = False,
	use_optimal_next_skill: bool = False,
):
	"""
	:param envs:
	:param low_policy:
	:param termination:
	:param target_skills: Length = n (# of envs), each has shape [M, d]
	:param termination_pred_interval:
	:param save_results:
	:param use_optimal_next_skill:
	:return:
	"""
	# Some variables

	n_envs = len(envs)
	subseq_len = envs[0].num_stack_frames
	semantic_skill_dim = termination.skill_dim

	n_possible_skills = target_skills.shape[1]  # TODO ????

	cur_skill_pos = np.array([0 for _ in range(n_envs)])  # [8, ]
	max_skills = np.array([n_possible_skills - 1 for _ in range(n_envs)])

	# Prepare save
	eval_infos = {f"env_{k}": defaultdict(list, env_name=envs[k].get_short_str_for_save()) for k in range(n_envs)}

	# Prepare env
	timestep = 0
	done = [False for _ in range(n_envs)]
	rew = np.array([0 for _ in range(n_envs)])

	obs_list = [envs[i].reset(target_skills[i][cur_skill_pos[i]]) for i in range(n_envs)]
	obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
	first_observations = obs["observations"][:, -1, ...].copy()
	returns = np.array([0.0 for _ in range(n_envs)])

	while not all(done):

		history_observations = obs["observations"]  # [8, 4, 140]
		history_actions = obs["actions"]  # [8, 4, 4]
		history_rewards = obs["rewards"]  # [8, 4]
		history_skills = obs["skills"]  # [8, 4, 512]
		history_maskings = obs["maskings"]
		timestep += 1

		done_prev = done.copy()

		if use_optimal_next_skill:
			cur_skill_pos = np.min([cur_skill_pos + rew, max_skills], axis=0)
			print("Cur skill pos", cur_skill_pos)
		else:
			if ((timestep - 1) % termination_pred_interval) == 0 and (timestep > 30):
				maybe_skill_done = termination.predict(
					observations=history_observations[:, -1, ...],  # Current observations
					first_observations=first_observations,
					skills=history_skills[:, -1, :semantic_skill_dim],
					binary=True
				)
				skill_done = np.where(maybe_skill_done == 1)[0]
				first_observations[skill_done] = history_observations[skill_done, -1, ...].copy()
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

		step_results = [env.step(act.copy(), cur_skills[i].copy()) for env, act, i in zip(envs, actions, range(n_envs))]
		obs_list = [result[I_OBS] for result in step_results]

		obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
		rew = np.stack([result[I_REWARD] for result in step_results])

		# if np.sum(rew) > 0:
		# 	print("POSITIVE!!!")
		# 	exit()
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

	print("=" * 30)
	n_tasks = sum([env.n_target for env in envs])
	print("Returns", returns)
	for (k, v), ret in zip(eval_infos.items(), returns):
		print(v["env_name"], ret)
	print(f"Total sum: {returns.sum()} among {len(envs)} tasks")
	print(f"Total success ratio: {100 * (returns.sum() / n_tasks)}%")
	print("=" * 30)
	return eval_infos
