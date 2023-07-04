from typing import Dict, List

from comde.baselines.prompt_dt import VLPromptDT
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class SourceLanguagePromptDT(VLPromptDT):
	"""
	SourceLanguagePromptDT: Video-Language prompt DT
	Input of PromptDT
		1. State-action-reward history (Like DT)
		2. Source skills & Language instructions as a prompt (Along 'sub-sequence' axis)

	This is also called Comde-GPT
	"""
	PARAM_COMPONENTS = ["policy"]

	def __str__(self):
		return "SourceLanguagePromptDT"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(SourceLanguagePromptDT, self).__init__(seed=seed, cfg=cfg, init_build_model=False)
		self.video_parsing = True
		self.prompt_dim = self.skill_dim

		if init_build_model:
			self.build_model()

	def get_prompts(self, replay_data: ComDeBufferSample):

		qkv_info = bert_base_forward(replay_data.language_guidance)

		prompts = qkv_info["language_embedding"]
		prompts_maskings = qkv_info["attention_mask"]
		return prompts, prompts_maskings

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
