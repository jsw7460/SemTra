from collections import defaultdict
from typing import List, Optional, Dict

import numpy as np
from jax.tree_util import tree_map

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.envs.utils import SkillHistoryEnv
from comde.utils.common.lang_representation import SkillRepresentation

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3

def evaluate_comde_batch(
	envs: List[SkillHistoryEnv],
	low_policy: BaseLowPolicy,
	seq2seq: BaseSeqToSeq,
	termination: BaseTermination,
	source_skills: List[np.ndarray],
	n_source_skills: List[int],
	language_guidance: List[np.ndarray],
	termination_pred_interval: int = 10,
	save_results: bool = False,
	skill_infos: Optional[Dict[str, List[SkillRepresentation]]] = None,
	use_optimal_target_skill: bool = False,
	use_optimal_next_skill: bool = False,
):
	"""
	:param envs:
	:param low_policy:
	:param seq2seq:
	:param termination:
	:param source_skills: Length = n (# of envs), each has shape [M, d]
	:param n_source_skills: Length = n (# of envs), Indicate the number of skills wo/ zero padding
	:param language_guidance: Length = n (# of envs), each has shape [d,]
	:param termination_pred_interval:
	:param save_results:
	:param use_optimal_next_skill:
	:param skill_infos:
	:return:
	"""
	# Some variables

	n_envs = len(envs)
	n_possible_skills = len(source_skills)

	subseq_len = envs[0].num_stack_frames

	# Get: Target skills
	source_skills = np.array(source_skills)  # [n, M, d]
	# source_skills = source_skills[:, :3, ...]
	n_source_skills = np.array(n_source_skills)
	language_guidances = np.array(language_guidance)  # [n, d]
	predictions = seq2seq.predict(
		source_skills=source_skills,
		language_operator=language_guidances,
		n_source_skills=n_source_skills
	)

	intents = predictions["pred_intents"]
	if use_optimal_target_skill:
		optimal_target_skills = []
		for env in envs:
			skill_list = env.skill_list
			optimal_target_skill = np.array([skill_infos[skill][0].vec for skill in skill_list])
			optimal_target_skills.append(optimal_target_skill)

		optimal_target_skills = np.array(optimal_target_skills)

		target_skills = optimal_target_skills
		intents = intents[:, :target_skills.shape[1], ...]

		print("Intents?", intents[0])


	else:
		target_skills = predictions["pred_skills_quantized"]	# [n, max_iter_len, d]

	semantic_skill_dim = target_skills.shape[-1]

	# intents = predictions["pred_intents"]
	#
	# # ===== Debug
	#
	# optimal_target_skills = []
	# for env in envs:
	# 	skill_list = env.skill_list
	# 	optimal_target_skill = np.array([skill_infos[skill][0].vec for skill in skill_list])
	# 	optimal_target_skills.append(optimal_target_skill)
	#
	# optimal_target_skills = np.array(optimal_target_skills)
	#
	# target_skills = optimal_target_skills
	#
	# pred_nearest_idxs = np.array(predictions["pred_nearest_idxs"])	# [max_skills, n_env], -2 for start-end tokens
	# pred_nearest_idxs = np.transpose(pred_nearest_idxs, (1, 0))[:, :3]	# [n_env, max_skills]
	#
	# gt_target_skill_idxs = []
	# for env in envs:
	# 	gt_target_skill_idxs.append(env.idx_skill_list)
	#
	# gt_target_skill_idxs = np.array(gt_target_skill_idxs)[:, :3]
	#
	# print("pred", pred_nearest_idxs)
	# print("target", gt_target_skill_idxs)
	# print(np.mean(pred_nearest_idxs == gt_target_skill_idxs))
	#
	# # return {}
	# # ===== Debug

	target_skills = np.concatenate((target_skills, intents), axis=-1)

	n_possible_skills = target_skills.shape[1]	# TODO ????

	cur_skill_pos = np.array([0 for _ in range(n_envs)])  # [8, ]
	max_skills = np.array([n_possible_skills - 1 for _ in range(n_envs)])

	# Prepare save
	eval_infos = {
		f"env_{k}": defaultdict(list, env_name=envs[k].get_short_str_for_save()) for k in range(n_envs)
	}

	# Prepare env
	timestep = 0
	done = [False for _ in range(n_envs)]
	rew = np.array([0 for _ in range(n_envs)])

	obs_list = [envs[i].reset(target_skills[i][cur_skill_pos[i]]) for i in range(n_envs)]
	obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
	first_observations = obs["observations"][:, -1, ...].copy()
	returns = np.array([0 for _ in range(n_envs)])

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
			if ((timestep - 1) % termination_pred_interval) == 0:
				maybe_skill_done = termination.predict(
					observations=history_observations[:, -1, ...],  # Current observations
					first_observations=first_observations,
					skills=history_skills[:, -1, :semantic_skill_dim],
					binary=True
				)
				cur_skill_pos = np.min([cur_skill_pos + maybe_skill_done, max_skills], axis=0)
				print("cur skill pos", cur_skill_pos)

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

	print("Returns", returns)
	for (k, v), ret in zip(eval_infos.items(), returns):
		print(v["env_name"], ret)
	return eval_infos
