from typing import Dict

import wandb
from stable_baselines3.common.logger import configure

from comde.utils.common.strings import mode_str


class Loggable:
	"""Not an interface"""
	WANDB_EXCLUDED = {"time/", "info/"}

	def __init__(self, cfg):
		self.wandb_logger = wandb.init(
			project=cfg["wandb"]["project"],
			entity=cfg["wandb"]["entity"],
			config=cfg,
			name=cfg["wandb"]["name"]
		)

		self.terminal_logger = configure(None, format_strings=["stdout"])
		self.wandb_url = wandb.run.get_url()

	def dump_logs(self, step: int):
		self.terminal_logger.record("info/step", step)
		self.wandb_logger.log(
			data={
				key: value for key, value in self.terminal_logger.name_to_value.items() \
				if not any([exc in key for exc in Loggable.WANDB_EXCLUDED])
			},
			step=step
		)
		self.terminal_logger.dump(step=step)
		# Too long...
		print(f" • Wandb url: {self.wandb_url}")

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
