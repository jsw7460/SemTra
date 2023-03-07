import gym
import numpy as np


obs_dim = 140
act_dim = 4
observation_space = gym.spaces.Box(shape=(obs_dim,), low=-np.inf, high=np.inf)
action_space = gym.spaces.Box(shape=(act_dim,), low=-1.0, high=1.0)
