from typing import Tuple, Dict, Any

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def demogen_policy_update(
	rng: jnp.ndarray,
	policy: Model,
	observations: jnp.ndarray,  # [b, l, d]
	video_embedding: jnp.ndarray,  # [b, d]	# Episodic video embedding
	non_functionality: jnp.ndarray,  # [b, d]
	parameters: jnp.ndarray,  # [b, l, d]	# Parameters of skill which should be excuted at the present.
	actions: jnp.ndarray,  # [b, l, d]
	maskings: jnp.ndarray  # [b, l]
):
	rng, dropout_key = jax.random.split(rng)
	subseq_len = actions.shape[1]
	action_dim = actions.shape[-1]

	video_embedding = jnp.expand_dims(video_embedding, axis=1)
	video_embedding = jnp.repeat(video_embedding, repeats=subseq_len, axis=1)

	non_functionality = jnp.expand_dims(non_functionality, axis=1)
	non_functionality = jnp.repeat(non_functionality, repeats=subseq_len, axis=1)

	nonfunc_param = jnp.concatenate((non_functionality, parameters), axis=-1)

	# Policy input:
	# 	1. observation
	# 	2. target video embedding
	#   3. non-functionality with parameters
	policy_input = jnp.concatenate((observations, video_embedding, nonfunc_param), axis=-1)

	target_actions = actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, Any]]:
		pred_actions = policy.apply_fn(
			{"params": params},
			policy_input,
			deterministic=False,
			rngs={"dropout": dropout_key}
		)
		pred_actions = pred_actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)

		mse_loss = jnp.sum(jnp.mean((pred_actions - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)

		_info = {"demogen/policy_loss": mse_loss}
		return mse_loss, _info

	new_policy, info = policy.apply_gradient(loss_fn=loss_fn)
	return new_policy, info


@jax.jit
def demogen_gravity_update(
	rng: jnp.ndarray,
	gravity: Model,
	source_video_embeddings: jnp.ndarray,  # [b, d] Episodic video embedding
	sequential_requirement: jnp.ndarray,  # [b, d]
	target_video_embedding: jnp.ndarray
):
	rng, dropout_key = jax.random.split(rng)
	gravity_input = jnp.concatenate((source_video_embeddings, sequential_requirement), axis=-1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, Any]]:
		pred_target_video = gravity.apply_fn(
			{"params": params},
			gravity_input,
			deterministic=False,
			rngs={"dropout_key": dropout_key}
		)
		gravity_loss = jnp.mean((pred_target_video - target_video_embedding) ** 2)

		return gravity_loss, {"demogen/generative_loss": gravity_loss, "__pred_target_video": pred_target_video}

	new_gravity, info = gravity.apply_gradient(loss_fn)

	return new_gravity, info
