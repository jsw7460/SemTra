import pickle
from typing import Dict, List

import jax.random
import numpy as np
import optax

from comde.baselines.algos.forwards import (
	demogen_gravity_forward as gravity_forward,
	demogen_policy_forward as policy_forward
)
from comde.baselines.algos.updates.bcz import (
	bcz_policy_update as policy_update,
	bcz_gravity_update as gravity_update
)
from comde.baselines.utils.get_episode_emb import get_video_text_embeddings
from comde.comde_modules.common.scaler import Scaler
from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class BCZ(BaseLowPolicy):
	"""
	Input of DemoGen
		1. observations
		2. source video & sequential requirement
		3. episodic instruction of target demonstrations (only during training)
		4. optimal parameter of skill which should be executed at the present
	3, 4 are big advantages compared to ComDe.

	Training:
		Input source video and sequential requirement (2) and apply contrastive or mse loss to minimize
		the distance with episodic instruction of target demonstrations (3)

		Then the policy input observations (1) and parameter (4).

	Evaluation:
		Input source video and sequential requirement.
		Using graivity model, predict the target episodic instruction.
		Then use it to predict the actions.
	"""

	PARAM_COMPONENTS = [
		"_BCZ__policy",
		"_BCZ__gravity"
	]

	def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
		super(BCZ, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

		episodic_inst_path = cfg["episodic_instruction_path"]
		with open(episodic_inst_path, "rb") as f:
			self.episodic_inst = pickle.load(f)

		videofeature_path = self.cfg["videofeature_path"]
		with open(videofeature_path, "rb") as f:
			self.video_feature = pickle.load(f)

		self.video_dim = self.cfg["video_dim"]
		self.language_dim = self.cfg["language_dim"]

		self.__gravity = None
		self.__policy = None

		if init_build_model:
			self.build_model()

	def __str__(self):
		return "BCZ"

	def _build_policy(self):
		policy_cfg = self.cfg["policy_cfg"].copy()
		lr = policy_cfg.pop("lr")
		policy = create_mlp(**policy_cfg)
		policy = Scaler(base_model=policy, scale=self.cfg["act_scale"])
		init_obs = np.zeros((1, self.observation_dim))
		init_inst = np.zeros((1, self.language_dim))
		init_params = np.zeros((1, self.nonfunc_dim + self.total_param_dim))
		policy_input = np.concatenate((init_obs, init_inst, init_params), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(learning_rate=lr)
		self.__policy = Model.create(model_def=policy, inputs=[rngs, policy_input], tx=tx)

	def _build_gravity(self):
		gravity_cfg = self.cfg["gravity_cfg"].copy()
		lr = gravity_cfg.pop("lr")
		gravity = create_mlp(**gravity_cfg)
		init_video = np.zeros((1, self.video_dim))
		init_seq = np.zeros((1, self.language_dim))
		init_gravity_input = np.concatenate((init_video, init_seq), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)
		tx = optax.adam(learning_rate=lr)
		self.__gravity = Model.create(model_def=gravity, inputs=[rngs, init_gravity_input], tx=tx)

	def build_model(self):
		self._build_gravity()
		self._build_policy()

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		video_text_info = get_video_text_embeddings(
			video_feature_dict=self.video_feature,
			text_dict=self.episodic_inst,
			replay_data=replay_data
		)
		source_video_embeddings = video_text_info["source_video_embeddings"]
		target_text_embs = video_text_info["target_text_embs"]

		new_gravity, gravity_info = gravity_update(
			rng=self.rng,
			gravity=self.__gravity,
			source_video_embeddings=source_video_embeddings,
			sequential_requirement=replay_data.sequential_requirement,
			episodic_instructions=target_text_embs
		)
		self.rng, _ = jax.random.split(self.rng)

		# Expand the parameter
		params_for_skills = np.repeat(replay_data.params_for_skills, axis=-1, repeats=self.param_repeats)

		episodic_inst = gravity_info.pop("__pred_episode_semantic")
		new_policy, policy_info = policy_update(
			rng=self.rng,
			policy=self.__policy,
			observations=replay_data.observations,
			episodic_inst=episodic_inst,
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
		sequential_requirement: np.ndarray,
		to_np: bool = True
	):
		gravity_input = np.concatenate((source_video_embeddings, sequential_requirement), axis=-1)
		self.rng, episodic_instruction = gravity_forward(rng=self.rng, model=self.__gravity, model_input=gravity_input)
		if to_np:
			return np.array(episodic_instruction)
		else:
			return episodic_instruction

	def predict_action(
		self,
		observations: np.ndarray,
		non_functionality: np.ndarray,
		current_params_for_skills: np.ndarray,
		episodic_instruction: np.ndarray,
		to_np: bool = True
	):
		policy_input = np.concatenate(
			(observations, episodic_instruction, non_functionality, current_params_for_skills),
			axis=-1
		)
		self.rng, actions = policy_forward(rng=self.rng, model=self.__policy, policy_input=policy_input)

		if to_np:
			return np.array(actions)
		else:
			return actions

	def predict(self, *args, **kwargs) -> np.ndarray:
		pass

	def evaluate(self, *args, **kwargs) -> Dict:
		return dict()

	@property
	def model(self) -> Model:
		return self.__policy

	def _excluded_save_params(self) -> List:
		return BCZ.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in BCZ.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return BCZ.PARAM_COMPONENTS
