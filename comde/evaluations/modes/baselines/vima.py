from collections import defaultdict
from typing import List, Dict

import numpy as np
from jax.tree_util import tree_map

from comde.baselines.prompt_dt import VLPromptDT
from comde.evaluations.utils.postprocess_evaldata import postprocess_eval_data as postprocess
from comde.rl.envs.utils import SkillHistoryEnv

I_OBS = 0
I_REWARD = 1
I_DONE = 2
I_INFO = 3


def evaluate_vima(
	envs: List[SkillHistoryEnv],
	baseline: VLPromptDT,
	prompts: Dict[str, np.ndarray],  # [n_envs, L, d]
	prompts_maskings: Dict[str, np.ndarray],
	sequential_requirement: np.ndarray = None,  # [n_envs, d]
	non_functionality: np.ndarray = None,  # [n_envs, d]
	param_for_skills: np.ndarray = None,  # [n_envs, n_source_skills, d]
	rtgs: np.ndarray = None,  # [n_envs,]
	save_results: bool = False
):
	# Some variables
	n_envs = len(envs)
	env_dummy_dim = envs[0].skill_dim
	subseq_len = envs[0].num_stack_frames

	dummy_parameterized_skills = np.zeros((env_dummy_dim,))

	# Prepare save
	eval_infos = {f"env_{k}": defaultdict(list, env_name=envs[k].get_short_str_for_save()) for k in range(n_envs)}

	# Prepare env
	timestep = 0
	done = [False for _ in range(n_envs)]
	# rew = np.array([0 for _ in range(n_envs)])

	obs_list = [envs[i].reset(dummy_parameterized_skills) for i in range(n_envs)]
	obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
	returns = np.array([0.0 for _ in range(n_envs)])

	if rtgs is not None:
		rtgs_for_concat = rtgs.copy()

		rtgs = rtgs[:, np.newaxis]
		rtgs = np.repeat(rtgs, repeats=subseq_len, axis=1)

	while not all(done):
		history_observations = obs["observations"]  # [8, 4, 140]
		history_actions = obs["actions"]  # [8, 4, 4]
		history_maskings = obs["maskings"]
		timestep += 1

		done_prev = done.copy()

		timesteps = np.arange(timestep - subseq_len, timestep)[np.newaxis, ...]
		timesteps = np.repeat(timesteps, axis=0, repeats=n_envs)
		timesteps[timesteps < 0] = -1

		actions = baseline.predict(
			observations=history_observations,
			actions=history_actions,
			rtgs=rtgs,
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			param_for_skills=param_for_skills,
			maskings=history_maskings,
			timesteps=timesteps,
			to_np=True,
			**prompts, **prompts_maskings
		)

		step_results = [
			env.step(act.copy(), dummy_parameterized_skills.copy())
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

		if rtgs is not None:
			rtgs_for_concat = rtgs_for_concat - rew.astype("float64")
			rtgs = np.concatenate((rtgs, rtgs_for_concat.reshape(-1, 1)), axis=-1)
			rtgs = rtgs[:, 1:]

		if save_results:
			for k in range(n_envs):
				eval_infos[f"env_{k}"]["observations"].append(obs_list[k])
				eval_infos[f"env_{k}"]["actions"].append(actions[k])
				eval_infos[f"env_{k}"]["rewards"].append(rew[k])
				eval_infos[f"env_{k}"]["infos"].append(step_results[k][I_INFO])
		else:
			for k in range(n_envs):
				eval_infos[f"env_{k}"]["rewards"].append(rew[k])

	n_tasks = sum([env.n_target for env in envs])
	return postprocess(eval_infos=eval_infos, n_tasks=n_tasks)
