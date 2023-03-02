from typing import Dict, Tuple, List

import gym
import numpy as np

from comde.comde_modules.low_policies.naive.decision_transformer import DecisionTransformer


def evaluate_primdt(
	env: gym.Env,
	dt: DecisionTransformer,
	init_rtgs: List,
	normalization_dict: Dict
) -> Tuple[Dict, Dict]:
	"""
		This function is not responsible for timelimit of environment.
	"""

	obs_mean = normalization_dict["obs_mean"]
	obs_std = normalization_dict["obs_std"]
	act_mean = normalization_dict["act_mean"]
	act_std = normalization_dict["act_std"]

	observation_dim = env.observation_space.shape[-1]
	action_dim = env.action_space.shape[-1]

	subseq_len = dt.cfg["subseq_len"]

	ret_info = {}

	# 첫 번째 action 얻을 때 -> 19개가 zero masking
	for rtg in init_rtgs:
		done = False
		timestep = 1

		observation = env.reset()

		episodic_observations = []
		episodic_termination_prediction = []
		episodic_primitive_actions = []
		episodic_rewards = []
		episodic_dones = []
		episodic_infos = []
		episodic_rtgs = []
		episodic_timesteps = []
		np_ep_actions = np.zeros((0, env.action_space.shape[-1]))
		while not done:
			episodic_observations.append(observation)
			episodic_rtgs.append(rtg - sum(episodic_rewards))
			episodic_timesteps.append(timestep - 1)

			# Make input of decision transformer
			np_ep_observations = np.vstack(episodic_observations)[-subseq_len:]
			np_ep_actions = np.vstack((np_ep_actions, np.zeros((1, action_dim))))[-subseq_len:]
			np_ep_timesteps = np.array(episodic_timesteps)[-subseq_len:]
			np_ep_rtgs = np.vstack(episodic_rtgs)[-subseq_len:]

			current_length = len(np_ep_observations)
			# Pad zeros to make input of dt
			dt_observations = np.concatenate(
				(np_ep_observations, np.zeros((subseq_len - np_ep_observations.shape[0], observation_dim))),
				axis=0
			).reshape(1, -1, observation_dim)
			dt_observations = (dt_observations - obs_mean) / obs_std

			dt_actions = np.concatenate(
				(np_ep_actions, np.zeros((subseq_len - np_ep_actions.shape[0], action_dim))),
				axis=0
			).reshape(1, -1, action_dim)
			dt_actions = (dt_actions - act_mean) / act_std

			dt_attention_mask = np.concatenate(
				(np.ones(current_length, ), np.zeros(subseq_len - current_length)),
			).reshape(1, -1)

			dt_timesteps = np.concatenate(
				(np_ep_timesteps, np.zeros((subseq_len - np_ep_timesteps.shape[0]))),
				axis=0
			).reshape(1, -1)

			dt_rtgs = np.concatenate(
				(np_ep_rtgs, np.zeros((subseq_len - np_ep_rtgs.shape[0], 1))),
				axis=0
			).reshape(1, -1, 1)

			primitive_action = dt.predict(
				observations=dt_observations,
				actions=dt_actions,
				timesteps=dt_timesteps.astype("i4"),
				maskings=dt_attention_mask,
				rtgs=dt_rtgs
			)
			primitive_action = np.array(primitive_action)
			np_ep_actions[-1] = primitive_action

			observation, reward, done, info = env.step(primitive_action.reshape(-1,))

			timestep += 1

			episodic_primitive_actions.append(primitive_action)
			episodic_rewards.append(reward)
			episodic_dones.append(done)
			episodic_infos.append(info)

		ret_info[f"target_{rtg}"] = {
			"observations": episodic_observations,
			"actions": episodic_primitive_actions,
			"rewards": episodic_rewards,
			"dones": episodic_dones,
			"infos": episodic_infos,
			"termination_predictions": episodic_termination_prediction
		}

	episodic_returns = [sum(ret_info[f"target_{rtg}"]["rewards"]) for rtg in init_rtgs]
	wandb_info = {"episodic_return": np.mean(episodic_returns).item()}

	return ret_info, wandb_info
