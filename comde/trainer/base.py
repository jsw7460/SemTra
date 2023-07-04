import pickle
from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Union, List, Type, Any

from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv
from comde.utils.superclasses.loggable import Loggable

SkillRelatedEnv = Union[List[SkillInfoEnv], List[Type], SkillInfoEnv, Dict[str, SkillInfoEnv], Any]


class BaseTrainer(Loggable):
	def __init__(self, cfg: Dict, env: Union[SkillInfoEnv, SkillRelatedEnv]):
		super(BaseTrainer, self).__init__(cfg=cfg)

		self.env = env

		# ==== time ====
		self.today = None
		self.start = None

		self.cfg = cfg
		self.prepare_run()

		self.n_update = 0
		self.batch_size = cfg["batch_size"]
		self.max_iter = cfg["max_iter"]
		self.step_per_dataset = cfg["step_per_dataset"]
		self.log_interval = cfg["log_interval"]
		self.save_interval = cfg["save_interval"]

		self.required_total_update = self.max_iter * self.step_per_dataset

	@abstractmethod
	def run(self, *args, **kwargs):
		"""Training loop"""

	@abstractmethod
	def save(self, *args, **kwargs):
		"""Save modules"""

	@abstractmethod
	def load(self, *args, **kwargs):
		"""Load modules"""

	def dump_logs(self, step: int):
		now = datetime.now()
		elapsed = (now - self.start).seconds
		fps = step / elapsed
		remain = int((self.required_total_update - step) / fps)
		eta = now + timedelta(seconds=remain)

		self.record({
			"time/fps": fps,
			"time/elapsed": str(timedelta(seconds=elapsed)),
			"time/remain": str(timedelta(seconds=remain)),
			"time/eta": eta.strftime("%m.%d / %H:%M:%S")
		})
		super(BaseTrainer, self).dump_logs(step=step)

	def prepare_run(self):
		prefix = self.cfg["save_prefix"]  # /home/jsw7460/comde_models
		suffix = self.cfg["save_suffix"]  # seq_hard
		self.today = datetime.today()
		today_str = self.today.strftime('%Y-%m-%d')  # 2023-03-06
		date_prefix = Path(prefix) / Path(today_str)  # /home/jsw7460/comde_models/2023-03-06

		cfg_prefix = (Path(date_prefix) / Path("cfg"))
		cfg_prefix.mkdir(parents=True, exist_ok=True)  # ~/cfg/

		self.cfg["save_paths"] = dict()

		for module_key in self.cfg["modules"]:
			module_prefix = (Path(date_prefix) / Path(f"{module_key}"))
			module_prefix.mkdir(parents=True, exist_ok=True)

			module_fullpath = module_prefix / Path(suffix)
			self.cfg["save_paths"][module_key] = str(module_fullpath)

		self.cfg.update({
			"date": today_str,
			"wandb_url": self.wandb_url
		})

		# Dump configure file
		with open(str(cfg_prefix / Path(f"cfg_{suffix}")), "wb") as f:
			pickle.dump(self.cfg, f)

		self.start = datetime.now()
