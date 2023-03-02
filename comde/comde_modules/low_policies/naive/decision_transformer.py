from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .architectures.decision_transformer import PrimDecisionTransformer
from comde.comde_modules.common.interfaces.i_savable import IJaxSavable
from comde.comde_modules.common.interfaces.i_trainable import ITrainable
from comde.comde_modules.low_policies.algos.forwards import primdecisiontransformer_forward as fwd
from comde.comde_modules.low_policies.algos.update_dt import dt_updt
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class DecisionTransformer(BaseLowPolicy, IJaxSavable, ITrainable):
	PARAM_COMPONENTS = ["_DecisionTransformer__model"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	):
		super(DecisionTransformer, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)
		self.__model = None
		self.optimizer = None
		self.scheduler = None
		self.max_ep_len = cfg["max_ep_len"]

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		return self.__model

	@model.setter
	def model(self, value):
		self.__model = value

	def _str_to_activation(self, activation_fn: str):
		return activation_fn

	def build_model(self):
		transformer = PrimDecisionTransformer(
			gpt2_config=self.cfg["gpt2_config"],
			obs_dim=self.observation_dim,
			act_dim=self.action_dim,
			hidden_size=self.cfg["hidden_size"],
			act_scale=self.cfg["act_scale"],
			max_ep_len=self.max_ep_len,
		)
		self.rng, rngs = get_basic_rngs(self.rng)
		self.rng, transformer_dropout = jax.random.split(self.rng)
		rngs.update({"transformer_dropout": transformer_dropout})
		subseq_len = self.cfg["subseq_len"]
		init_observations = np.zeros((1, subseq_len, self.observation_dim))
		init_actions = np.zeros((1, subseq_len, self.action_dim))
		init_timesteps = np.zeros((1, self.cfg["subseq_len"]), dtype="i4")
		init_masks = np.zeros((1, self.cfg["subseq_len"]))
		init_rtgs = np.zeros((1, subseq_len, 1))

		tx = optax.chain(
			optax.clip(1.0),
			optax.adamw(learning_rate=self.cfg["lr"])
		)

		self.model = Model.create(
			model_def=transformer,
			inputs=[
				rngs,
				init_observations,
				init_actions,
				init_timesteps,
				init_masks,
				init_rtgs,
			],
			tx=tx
		)

	def update(self, replay_data: ComDeBufferSample) -> Dict:
		new_model, info = dt_updt(
			rng=self.rng,
			dt=self.model,
			observations=replay_data.observations,
			actions=replay_data.actions,
			rtgs=replay_data.rtgs[..., np.newaxis],
			timesteps=replay_data.timesteps.astype("i4"),
			maskings=replay_data.maskings,
			action_targets=jnp.copy(replay_data.actions),
		)

		self.model = new_model
		self.rng, _ = jax.random.split(self.rng)
		self.record_mean(info)

		return info

	def predict(
		self,
		observations: jnp.ndarray,
		actions: jnp.ndarray,
		timesteps: jnp.ndarray,
		maskings: Union[jnp.ndarray],
		rtgs: jnp.ndarray = None,
	) -> jnp.ndarray:
		self.rng, prediction = fwd(
			rng=self.rng,
			model=self.model,
			observations=observations,
			actions=actions,
			timesteps=timesteps,
			maskings=maskings,
			rtgs=rtgs,
		)
		obs_preds, action_preds, return_preds = prediction
		return action_preds[:, -1]

	def evaluate(
		self,
		observations: jnp.ndarray,
		actions: jnp.ndarray,
		maskings: jnp.ndarray,
		**kwargs
	) -> Dict:
		if observations.ndim == 2:
			observations = observations[np.newaxis, ...]
			actions = actions[np.newaxis, ...]
			maskings = maskings[np.newaxis, ...]
		action_preds = self.predict(
			observations=observations,
			actions=actions,
			maskings=maskings,
		)
		action_preds = action_preds.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)
		action_targets = actions.reshape(-1, self.action_dim) * maskings.reshape(-1, 1)

		mse_error = (action_preds - action_targets) ** 2
		mse_error = jnp.mean(mse_error)

		eval_info = {
			"decoder/mse_error": mse_error,
			"decoder/mse_error_scaled(x100)": mse_error * 100
		}
		return eval_info

	def _excluded_save_params(self) -> List:
		return DecisionTransformer.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in DecisionTransformer.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return DecisionTransformer.PARAM_COMPONENTS
