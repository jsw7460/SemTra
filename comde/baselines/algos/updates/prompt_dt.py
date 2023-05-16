from typing import Tuple, Dict, Any

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def promptdt_update(
	rng: jnp.ndarray,
	policy: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,  # [b, l, d]
	rtgs: jnp.ndarray,  # [b, l, 1]
	prompts: jnp.ndarray,  # [b, M, d]
	prompts_maskings: jnp.ndarray,  # [b, M]: indicate whether dummy source skills
	sequential_requirement: jnp.ndarray,  # [b, d]
	non_functionality: jnp.ndarray,  # [b, d]
	param_for_skills: jnp.ndarray,  # [b, M, d]
	timesteps: jnp.ndarray,  # [b, l]
	maskings: jnp.ndarray,  # [b, l]
) -> Tuple[Model, Dict[str, Any]]:
	rng, dropout_key = jax.random.split(rng)
	action_dim = actions.shape[-1]

	target_actions = actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, Any]]:
		action_preds = policy.apply_fn(
			{"params": params},
			observations=observations,
			actions=actions,
			rtgs=rtgs,
			prompts=prompts,
			prompts_maskings=prompts_maskings,
			sequential_requirement=sequential_requirement,
			non_functionality=non_functionality,
			param_for_skills=param_for_skills,
			timesteps=timesteps,
			maskings=maskings,
			rngs={"dropout": dropout_key}
		)

		action_preds = action_preds.reshape(-1, action_dim) * maskings.reshape(-1, 1)

		action_loss = jnp.sum(jnp.mean((action_preds - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)

		_infos = {"promptdt/bc_loss": action_loss}

		return action_loss, _infos

	new_policy, info = policy.apply_gradient(loss_fn)

	return new_policy, info
