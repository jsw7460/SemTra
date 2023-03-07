import gym


class DimensionFix(gym.Wrapper):
	def __init__(self, env: gym.Env, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
		super(DimensionFix, self).__init__(env=env)
		self.observation_space = observation_space
		self.action_space = action_space
