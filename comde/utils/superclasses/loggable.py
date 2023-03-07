from typing import Dict

from stable_baselines3.common.logger import configure

from comde.utils.common.strings import mode_str


class Loggable:
	"""Not an interface"""

	def __init__(self):
		self.terminal_logger = configure(None, format_strings=["stdout"])

	def dump_logs(self, step: int):
		self.terminal_logger.record("info/step", step)
		self.terminal_logger.dump(step=step)

	def record(self, log_dict: Dict):
		for key, value in log_dict.items():
			self.terminal_logger.record(key, value)

	def record_mean(self, log_dict: Dict):
		for key, value in log_dict.items():
			self.terminal_logger.record_mean(key, value)

	def record_from_dicts(self, *args: Dict, mode: str):
		"""
		:param mode: Record will be logged like "mode/...loss, ..."
		"""
		records = dict()
		[records.update(record) for record in args]
		records = {mode_str(mode, key): value for key, value in records.items() if not "__" in key}
		self.record_mean(records)
