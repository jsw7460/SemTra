from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def dt_updt(
	rng: jnp.ndarray,
	dt: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	rtgs: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	action_targets: jnp.ndarray,
):
	rng, dropout_key = jax.random.split(rng)

	action_dim = actions.shape[-1]
	action_targets = action_targets.reshape(-1, action_dim) * maskings.reshape(-1, 1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		predictions = dt.apply_fn(
			{"params": params},
			observations=observations,
			actions=actions,
			timesteps=timesteps,
			maskings=maskings,
			rtgs=rtgs,
			deterministic=False,
			rngs={"dropout": dropout_key},
		)
		observation_preds, action_preds, ret_preds = predictions
		action_preds = action_preds.reshape(-1, action_dim) * maskings.reshape(-1, 1)
		action_loss = jnp.mean((action_preds - action_targets) ** 2)

		loss = action_loss
		_infos = {
			"skill_decoder/mse_loss": loss,
		}
		return loss, _infos

	new_dt, infos = dt.apply_gradient(loss_fn)
	return new_dt, infos
