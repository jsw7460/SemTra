import gym


class StringMetaWorld(gym.Wrapper):
	def __init__(self, env: gym.Env):
		super(StringMetaWorld, self).__init__(env=env)

	def get_short_str_for_save(self) -> str:
		return "".join([skill[0] for skill in self.env.skill_list])

	def __str__(self):
		return self.get_short_str_for_save()
