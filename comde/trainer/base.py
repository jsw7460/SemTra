from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict
import pickle

from comde.utils.superclasses.loggable import Loggable


class BaseTrainer(Loggable):
	def __init__(self, cfg: Dict):
		super(BaseTrainer, self).__init__()

		self.cfg = cfg
		self.prepare_run()

		self.batch_size = cfg["batch_size"]
		self.step_per_dataset = cfg["step_per_dataset"]

	@abstractmethod
	def run(self):
		"""Training loop"""

	@abstractmethod
	def save(self, *args, **kwargs):
		"""Save modules"""

	@abstractmethod
	def load(self, *args, **kwargs):
		"""Load modules"""

	def prepare_run(self):
		prefix = self.cfg["save_prefix"]  # /home/jsw7460/comde_models
		postfix = self.cfg["save_postfix"]		# seq_hard
		today_str = datetime.today().strftime('%Y-%m-%d')  # 2023-03-06
		date_prefix = Path(prefix) / Path(today_str)  # /home/jsw7460/comde_models/2023-03-06

		cfg_prefix = (Path(date_prefix) / Path("cfg"))
		cfg_prefix.mkdir(parents=True, exist_ok=True)  # ~/cfg/

		self.cfg["save_paths"] = dict()

		for module_key in self.cfg["modules"]:
			module_prefix = (Path(date_prefix) / Path(f"{module_key}"))
			module_prefix.mkdir(parents=True, exist_ok=True)

			module_fullpath = module_prefix / Path(postfix)
			self.cfg["save_paths"][module_key] = str(module_fullpath)

		self.cfg.update({"date": today_str})
		with open(str(cfg_prefix / Path(f"cfg_{postfix}")), "wb") as f:
			pickle.dump(self.cfg, f)
