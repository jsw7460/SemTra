import pickle
from typing import Dict, List

import numpy as np
import optax

from comde.baselines.algos.forwards import (
	demogen_gravity_forward as gravity_forward,
	demogen_policy_forward as policy_forward
)
from comde.baselines.algos.updates.demogen import (
	demogen_policy_update as policy_update,
	demogen_gravity_update as gravity_update
)
from comde.baselines.utils.get_episode_emb import get_video_text_embeddings
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class DemoGen(BaseLowPolicy):
	"""
	Input of DemoGen
		1. observations
		2. Source video & sequential requirement
		3. optimal parameter of skill which should be executed at the present

	See that 3. is big advantage for this model. (Because we assume environment tells optimal skills)
	The source video and sequential requirement (2.) are concatenated and embedded to the target video embedding.
	This target video embedding is conditioned to the policy.
	"""

	PARAM_COMPONENTS = [
		"_DemoGen__policy",
		"_DemoGen__gravity"
	]

	def __str__(self):
		return "DemoGen"

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(DemoGen, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)
		self.video_parsing = True
		self.video_dim = self.cfg["video_dim"]  # Not a raw embedding. Embeddings of pre-trained model.
		self.language_dim = self.cfg["language_dim"]  # Not a raw dimension. Embeddings of pre-trained model.

		self.__policy = None
		self.__gravity = None

		self.episodic_inst = None
		self.video_feature_dict = None

		if init_build_model:
			self.build_model()

	def _set_dictionaries(self):
		episodic_inst_path = self.cfg["episodic_instruction_path"]
		with open(episodic_inst_path, "rb") as f:
			self.episodic_inst = pickle.load(f)

		videofeature_path = self.cfg["videofeature_path"]
		with open(videofeature_path, "rb") as f:
			self.video_feature_dict = pickle.load(f)

	def build_model(self):
		self._set_dictionaries()
		self._build_policy()
		self._build_gravity()

	def _build_policy(self):
		policy_cfg = self.cfg["policy_cfg"].copy()
		policy_lr = policy_cfg.pop("lr")
		policy = create_mlp(**policy_cfg)
		policy = Scaler(base_model=policy, scale=np.array(self.cfg["act_scale"]))
		init_obs = np.zeros((1, self.observation_dim))
		init_demo = np.zeros((1, self.video_dim))
		init_params = np.zeros((1, self.nonfunc_dim + self.total_param_dim))
		policy_input = np.concatenate((init_obs, init_demo, init_params), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(policy_lr)
		self.__policy = Model.create(model_def=policy, inputs=[rngs, policy_input], tx=tx)

	def _build_gravity(self):
		gravity_cfg = self.cfg["gravity_cfg"].copy()
		gravity_lr = gravity_cfg.pop("lr")
		gravity = create_mlp(**gravity_cfg)
		init_video = np.zeros((1, self.video_dim))
		init_language = np.zeros((1, self.language_dim))
		gravity_input = np.concatenate((init_video, init_language), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(gravity_lr)
		self.__gravity = Model.create(model_def=gravity, inputs=[rngs, gravity_input], tx=tx)

	def get_source_target_info(self, replay_data: ComDeBufferSample):
		info = get_video_text_embeddings(
			video_feature_dict=self.video_feature_dict,
			text_dict=self.episodic_inst,
			replay_data=replay_data
		)

		ret_info = {
			"source": info["source_video_embeddings"],
			'target': info["target_video_embeddings"]
		}
		return ret_info

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		info = self.get_source_target_info(replay_data)
		source_video_embeddings = info["source"]
		target_video_embedding = info["target"]

		new_gravity, gravity_info = gravity_update(
			rng=self.rng,
			gravity=self.__gravity,
			source_video_embeddings=source_video_embeddings,
			sequential_requirement=replay_data.sequential_requirement,
			target_video_embedding=target_video_embedding
		)
		# Expand the parameter
		params_for_skills = np.repeat(replay_data.params_for_skills, axis=-1, repeats=self.param_repeats)

		video_embedding = gravity_info.pop("__pred_target_video")
		new_policy, policy_info = policy_update(
			rng=self.rng,
			policy=self.__policy,
			observations=replay_data.observations,
			video_embedding=video_embedding,
			non_functionality=replay_data.non_functionality,
			parameters=params_for_skills,
			actions=replay_data.actions,
			maskings=replay_data.maskings
		)
		self.__gravity = new_gravity
		self.__policy = new_policy
		return {**gravity_info, **policy_info}

	def predict_demo(
		self,
		source_video_embeddings: np.ndarray,  # [b, d]
		sequential_requirement: np.ndarray,  # [b, d]
		to_np: bool = True
	):
		gravity_input = np.concatenate((source_video_embeddings, sequential_requirement), axis=-1)
		self.rng, target_demo = gravity_forward(rng=self.rng, model=self.__gravity, model_input=gravity_input)
		if to_np:
			return np.array(target_demo)
		else:
			return target_demo

	def predict_action(
		self,
		observations: np.ndarray,
		non_functionality: np.ndarray,
		current_params_for_skills: np.ndarray,  # [b, d]. Parameter for the current skills to be excuted, thus 2dim.
		target_demo: np.ndarray,
		to_np: bool = True
	):
		policy_input = np.concatenate(
			(observations, target_demo, non_functionality, current_params_for_skills),
			axis=-1
		)
		self.rng, actions = policy_forward(rng=self.rng, model=self.__policy, policy_input=policy_input)

		if to_np:
			return np.array(actions)
		else:
			return actions

	def predict(self, *args, **kwargs) -> None:
		raise NotImplementedError("Use predict_demo or predict_action.")

	def evaluate(self, *args, **kwargs) -> Dict:
		"""..."""

	@property
	def model(self) -> Model:
		return self.__policy

	def _excluded_save_params(self) -> List:
		return DemoGen.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in DemoGen.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return DemoGen.PARAM_COMPONENTS
