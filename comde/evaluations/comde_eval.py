from typing import Dict, Tuple, List

import gym
import numpy as np

from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.comde_modules.seq2seq.base import BaseSeqToSeq
from comde.comde_modules.termination.base import BaseTermination


def evaluate_comde(
	env: gym.Env,
	low_policy: BaseLowPolicy,
	seq2seq: BaseSeqToSeq,
	termination: BaseTermination,
	source_skills: np.ndarray,	# [1, M, d]
	language_guidence: np.ndarray	# [1, d]
):

	observation_dim = env.observation_space.shape[-1]	# type: int
	action_dim = env.action_space.shape[-1]	# type: int
	skill_dim = low_policy.skill_dim

	cur_skill_pos = 0

	timestep = 0
	info = dict()

	done = False
	###
	target_skills = seq2seq.predict(
		source_skills=source_skills,
		language_operator=language_guidence
	)  # [b, max_iter_len, d]
	n_max_skills = target_skills.shape[1]

	ep_observations = []
	ep_actions = []
	ep_skills = []
	ep_timesteps = []
	ep_rewards = []
	ep_dones = []
	ep_infos = []

	observation = env.reset()
	action = np.zeros((0, action_dim))
	skill = target_skills[:, cur_skill_pos]

	ep_observations.append(observation.copy())
	ep_actions.append(action.copy())
	ep_skills.append(skill.copy())
	ep_timesteps.append(timestep)

	while not done:
		input_observations = np.array(ep_observations)
		input_timesteps = np.array(ep_timesteps)

		ep_actions.append(action)

		input_actions = np.concatenate((np.concatenate(ep_actions, axis=0), np.zeros((1, action_dim))), axis=0)
		input_skills = np.concatenate((np.concatenate(ep_skills, axis=0), np.zeros((1, skill_dim))), axis=0)

		action = low_policy.predict(
			observations=input_observations.reshape(1, -1, observation_dim),
			actions=input_actions.reshape(1, -1, action_dim),
			skills=input_skills.reshape(1, -1, skill_dim),
			timesteps=input_timesteps.reshape(1, -1),
			to_np=True
		)
		observation, reward, done, info = env.step(action.reshape(-1,).copy())

		timestep += 1

		ep_observations.append(observation.copy())
		ep_skills.append(target_skills[:, cur_skill_pos].copy())
		ep_timesteps.append(timestep)
		ep_actions[-1] = action.reshape(1, -1)
