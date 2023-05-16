from typing import Tuple, Dict, Any

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


def flatbc_update(
	rng: jnp.ndarray,
	policy: Model,
	observations: jnp.ndarray,	# [b, l, d]
	target_skills: jnp.ndarray,	# [b, M, d], M = the number of target skills (Fixed)
	target_skills_order: jnp.ndarray,	# [b, l]
	actions: jnp.ndarray,	# [b, l, d]
	maskings: jnp.ndarray	# [b, l]
):
	rng, dropout_key = jax.random.split(rng)
	batch_size = actions.shape[0]
	subseq_len = actions.shape[1]
	action_dim = actions.shape[-1]

	maskings = maskings.reshape(-1, 1)
	target_actions = actions.reshape(-1, action_dim) * maskings

	target_skills = target_skills.reshape(batch_size, -1)
	target_skills = jnp.expand_dims(target_skills, axis=1)
	target_skills = jnp.repeat(target_skills, repeats=subseq_len, axis=1)	# [b, l, d]

	target_skills_order = target_skills_order[..., jnp.newaxis]
	policy_input = jnp.concatenate((observations, target_skills, target_skills_order), axis=-1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, Any]]:

		pred_actions = policy.apply_fn(
			{"params": params},
			x=policy_input,
			deterministic=False,
			training=True,
			rngs={"dropout": dropout_key}
		)

		pred_actions = pred_actions.reshape(-1, action_dim) * maskings

		_loss = jnp.sum(jnp.mean((pred_actions - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)
		return _loss, {"flatbc/mse_loss": _loss}

	new_policy, info = policy.apply_gradient(loss_fn)
	return new_policy, info