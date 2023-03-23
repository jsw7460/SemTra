from collections import defaultdict
from typing import List

import numpy as np
from jax.tree_util import tree_map

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination
from comde.rl.envs.utils import SkillHistoryEnv

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3


# def evaluate_comde(
# 	env: gym.Env,
# 	low_policy: BaseLowPolicy,
# 	seq2seq: BaseSeqToSeq,
# 	termination: BaseTermination,
# 	source_skills: np.ndarray,  # [1, M, d]
# 	language_guidance: np.ndarray,  # [1, d]
# 	termination_pred_interval: int = 10,
# 	use_optimal_next_skill: bool = True
# ):
# 	observation_dim = env.observation_space.shape[-1]  # type: int
# 	action_dim = env.action_space.shape[-1]  # type: int
# 	skill_dim = low_policy.skill_dim
#
# 	cur_skill_pos = 0
#
# 	timestep = 0
# 	info = dict()
#
# 	done = False
# 	source_skills = source_skills.reshape(1, -1, source_skills.shape[-1])
# 	language_guidance = language_guidance.reshape(1, language_guidance.shape[-1])
#
# 	target_skills = seq2seq.predict(
# 		source_skills=source_skills,
# 		language_operator=language_guidance
# 	)  # [1, max_iter_len, d]
#
# 	target_skills = source_skills  # Debugging
# 	n_max_skills = target_skills.shape[1]
#
# 	ep_observations = []
# 	ep_actions = []
# 	ep_skills = []
# 	ep_timesteps = []
# 	ep_rewards = []
# 	ep_dones = []
# 	ep_infos = []
#
# 	observation = env.reset()
# 	first_obs_of_skill = observation.copy()
# 	action = np.zeros((0, action_dim))
# 	skill = target_skills[:, cur_skill_pos]
#
# 	ep_observations.append(observation.copy())
# 	ep_actions.append(action.copy())
# 	ep_skills.append(skill.copy())
# 	ep_timesteps.append(timestep)
#
# 	while not done:
# 		input_observations = np.array(ep_observations)
# 		input_timesteps = np.array(ep_timesteps)
#
# 		ep_actions.append(action)
#
# 		input_actions = np.concatenate((np.concatenate(ep_actions, axis=0), np.zeros((1, action_dim))), axis=0)
# 		input_skills = np.concatenate((np.concatenate(ep_skills, axis=0), np.zeros((1, skill_dim))), axis=0)
#
# 		if use_optimal_next_skill:
# 			maybe_skill_done = len(ep_rewards) > 0 and ep_rewards[-1] > 0
# 			if maybe_skill_done:
# 				print("Maybe skill done")
# 				cur_skill_pos = min(cur_skill_pos + 1, n_max_skills - 1)
# 				skill = target_skills[:, cur_skill_pos].copy()
#
# 		else:
# 			if ((timestep - 1) % termination_pred_interval) == 0:
# 				maybe_skill_done = termination.predict(
# 					observations=observation.reshape(1, -1, observation_dim),
# 					first_observations=first_obs_of_skill.reshape(1, -1, observation_dim),
# 					skills=skill.reshape(1, -1, skill_dim),
# 					binary=True
# 				)
# 				if maybe_skill_done:
# 					cur_skill_pos = min(cur_skill_pos + 1, n_max_skills - 1)
# 					skill = target_skills[:, cur_skill_pos].copy()
#
# 		action = low_policy.predict(
# 			observations=input_observations.reshape(1, -1, observation_dim),
# 			actions=input_actions.reshape(1, -1, action_dim),
# 			skills=input_skills.reshape(1, -1, skill_dim),
# 			timesteps=input_timesteps.reshape(1, -1),
# 			to_np=True
# 		)
# 		observation, reward, done, info = env.step(action.reshape(-1, ).copy())
# 		timestep += 1
#
# 		ep_observations.append(observation.copy())
# 		ep_skills.append(skill.copy())
# 		ep_timesteps.append(timestep)
# 		ep_actions[-1] = action.reshape(1, -1)
#
# 		ep_rewards.append(reward)
# 		ep_dones.append(done)
# 		ep_infos.append(info)
#
# 	print("Rewards:", sum(ep_rewards))
# 	return {
# 		"ep_observations": ep_observations,
# 		"ep_actions": ep_actions,
# 		"ep_skills": ep_skills,
# 		"ep_timesteps": ep_timesteps,
# 		"ep_rewards": ep_rewards,
# 		"ep_dones": ep_dones,
# 		"ep_infos": ep_infos,
# 		"return": sum(ep_rewards)
# 	}


def evaluate_comde_batch(
	envs: List[SkillHistoryEnv],
	low_policy: BaseLowPolicy,
	seq2seq: BaseSeqToSeq,
	termination: BaseTermination,
	source_skills: List[np.ndarray],  # Length = n (# of envs), each has shame [M, d]
	language_guidance: List[np.ndarray],  # Length = n (# of envs), each has shame [d,]
	termination_pred_interval: int = 10,
	save_results: bool = False,
	use_optimal_next_skill: bool = True,
):
	# Some variables
	n_envs = len(envs)
	n_possible_skills = len(source_skills)

	# Prepare save
	eval_infos = {
		f"env_{k}": defaultdict(list, env_name=envs[k].get_short_str_for_save()) for k in range(n_envs)
	}

	# Get: Target skills
	source_skills = np.array(source_skills)  # [n, M, d]
	language_guidances = np.array(language_guidance)  # [n, d]
	target_skills = seq2seq.predict(
		source_skills=source_skills,
		language_operator=language_guidances
	)  # [n, max_iter_len, d]
	cur_skill_pos = np.array([0 for _ in range(n_envs)])  # [8, ]
	max_skills = np.array([n_possible_skills - 1 for _ in range(n_envs)])

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

		done_prev = done

		if use_optimal_next_skill:
			cur_skill_pos = np.max([cur_skill_pos + rew, max_skills], axis=0)
		else:
			if ((timestep - 1) % termination_pred_interval) == 0:
				maybe_skill_done = termination.predict(
					observations=history_observations[:, -1, ...],  # Current observations
					first_observations=first_observations,
					skills=history_skills[:, -1, ...],
					binary=True
				)
				cur_skill_pos = np.max([cur_skill_pos + maybe_skill_done, max_skills], axis=0)

		cur_skills = target_skills[np.arange(n_envs), cur_skill_pos, ...]

		actions = low_policy.predict(
			observations=history_observations,
			actions=history_actions,
			skills=history_skills,
			maskings=history_maskings,
			timesteps=np.zeros_like(history_rewards),  # Note: In Comde, we dont use timesteps.
			to_np=True
		)

		step_results = [env.step(act, cur_skills[i]) for env, act, i in zip(envs, actions, range(n_envs))]
		obs_list = [result[I_OBS] for result in step_results]

		obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
		rew = np.stack([result[I_REWARD] for result in step_results])
		if np.sum(rew) > 0:
			print("POSITIVE!!!")
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
		rew = rew * (1 - done)
		returns += rew

	return eval_infos
