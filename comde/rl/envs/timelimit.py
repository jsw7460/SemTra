import gym


class TimeLimitEnv(gym.Wrapper):
	def __init__(self, env: gym.Env, limit: int):
		super(TimeLimitEnv, self).__init__(env)
		self.timestep = 0
		self.limit = limit

	def reset(self, **kwargs):
		self.timestep = 0
		return super(TimeLimitEnv, self).reset(**kwargs)

	def step(self, action):
		obs, reward, done, info = super(TimeLimitEnv, self).step(action)
		self.timestep += 1
		if self.timestep == self.limit:
			done = True

		return obs, reward, done, info