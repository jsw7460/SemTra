from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def skill_mlp_updt(
	rng: jnp.ndarray,
	mlp: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	maskings: jnp.ndarray
):
	rng, dropout_key = jax.random.split(rng)
	action_dim = actions.shape[-1]
	if maskings is None:
		maskings = jnp.ones(actions.shape[0])

	actions = actions.reshape(-1, action_dim)
	maskings = maskings.reshape(-1, 1)
	target_actions = actions * maskings

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		pred_actions = mlp.apply_fn(
			{"params": params},
			observations=observations,
			skills=skills,
			rngs={"dropout": dropout_key},
			deterministic=False,
			training=True
		)
		pred_actions = pred_actions.reshape(-1, action_dim) * maskings
		mse_loss = jnp.sum(jnp.mean((pred_actions - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)

		_infos = {
			"skill_decoder/mse_loss": mse_loss,
			"__decoder/pred_actions": pred_actions,
			"__decoder/target_actions": target_actions
		}
		return mse_loss, _infos

	new_mlp, infos = mlp.apply_gradient(loss_fn)
	return new_mlp, infos
