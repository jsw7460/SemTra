from stable_baselines3.common.logger import configure
from typing import Dict


class Loggable:
	"""Not an interface"""
	def __init__(self):
		self.terminal_logger = configure(None, format_strings=["stdout"])

	def dump_logs(self, step: int):
		self.terminal_logger.record("info/step", step)
		self.terminal_logger.dump(step=step)

	def record_mean(self, log_dict: Dict):
		for key, value in log_dict.items():
			self.terminal_logger.record_mean(key, value)
