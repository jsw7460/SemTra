from abc import abstractmethod
from typing import Dict

from comde.utils.superclasses.loggable import Loggable


class BaseTrainer(Loggable):
	def __init__(self, cfg: Dict):
		super(BaseTrainer, self).__init__()
		self.n_data_iter = cfg["n_data_iter"]
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
