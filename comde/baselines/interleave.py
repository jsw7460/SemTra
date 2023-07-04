import pickle
import random
from typing import Dict, List
from collections import defaultdict

from comde.baselines.prompt_dt import VLPromptDT
from comde.baselines.utils.interleave_template import template
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation


class InterleavedPromptDT(VLPromptDT):
	"""
	InterleavedPromptDT: Video-Language prompt DT
	Input of PromptDT
		1. State-action-reward history (Like DT)
		2. Source skills & Language instructions are INTERLEAVED as a prompt (Along 'sub-sequence' axis).
	"""
	PARAM_COMPONENTS = ["policy"]

	def __str__(self):
		return "InterleavedPromptDT"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(InterleavedPromptDT, self).__init__(seed=seed, cfg=cfg, init_build_model=False)
		self.prompt_dim = self.skill_dim
		del self.firstimage_mapping  # Not required for this class

		with open(cfg["skill_infos_path"], "rb") as f:
			skill_infos = pickle.load(f)	# type: Dict[str, List[SkillRepresentation]]

		index_to_skills = defaultdict(list)
		for key, val in skill_infos.items():
			index_to_skills[str(val[0].index)] = val

		self.skill_infos = skill_infos
		self.index_to_skills = index_to_skills	# type: Dict[str, List[SkillRepresentation]]

		if init_build_model:
			self.build_model()

	def get_prompts(self, replay_data: ComDeBufferSample):
		str_sequential_requirement = replay_data.str_sequential_requirement
		str_non_functionality = replay_data.str_non_functionality
		n_source_skills = replay_data.n_source_skills
		source_skills_idxs = replay_data.source_skills_idxs
		parameters = replay_data.parameters

		for seq_req, nf, n_sk, source_skills, parameter in zip(
			str_sequential_requirement, str_non_functionality, n_source_skills, source_skills_idxs, parameters
		):
			placeholder = random.choice(template[nf])
			source_skills = source_skills[:n_sk]
			video_fill = ", then ".join(
				[random.choice(self.index_to_skills[str(sk)]).variation for sk in source_skills]
			)
			seq_req_fill = seq_req
			skill_fill, param_fill = random.choice(list(parameter.items()))

			# Todo: From here~!

			print("seq req", seq_req_fill)
			print("video fill", video_fill)
			print("skill fill", skill_fill)
			print("param fill", param_fill)

		return None

	def evaluate(self, *args, **kwargs) -> Dict:
		return dict()

	@property
	def model(self) -> Model:
		return self.policy

	def _excluded_save_params(self) -> List:
		return InterleavedPromptDT.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in InterleavedPromptDT.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return InterleavedPromptDT.PARAM_COMPONENTS
