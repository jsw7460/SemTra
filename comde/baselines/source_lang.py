from typing import Dict, List

import numpy as np

from comde.baselines.prompt_dt import VLPromptDT
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SourceLanguagePromptDT(VLPromptDT):
	"""
	SourceLanguagePromptDT: Video-Language prompt DT
	Input of PromptDT
		1. State-action-reward history (Like DT)
		2. Source skills & Language instructions as a prompt (Along 'sub-sequence' axis)
	"""
	PARAM_COMPONENTS = ["policy"]

	def __str__(self):
		return "SourceLanguagePromptDT"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(SourceLanguagePromptDT, self).__init__(seed=seed, cfg=cfg, init_build_model=False)
		self.prompt_dim = self.skill_dim
		del self.firstimage_mapping  # Not required for this class

		if init_build_model:
			self.build_model()

	def get_prompts(self, replay_data: ComDeBufferSample):
		source_skills = replay_data.source_skills
		n_source_skills = replay_data.n_source_skills.reshape(-1, 1)
		batch_size = source_skills.shape[0]
		source_skills_maskings = np.arange(source_skills.shape[1]).reshape(1, -1)  # [1, M]
		source_skills_maskings = np.repeat(source_skills_maskings, repeats=batch_size, axis=0)  # [b, M]
		source_skills_maskings = np.where(source_skills_maskings < n_source_skills, 1, 0)
		return source_skills, source_skills_maskings

	def evaluate(self, *args, **kwargs) -> Dict:
		return dict()

	@property
	def model(self) -> Model:
		return self.policy

	def _excluded_save_params(self) -> List:
		return SourceLanguagePromptDT.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in SourceLanguagePromptDT.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return SourceLanguagePromptDT.PARAM_COMPONENTS
