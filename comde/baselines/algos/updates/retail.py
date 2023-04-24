from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def retail_policy_update(
	rng: jnp.ndarray,
	policy: Model,
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
		pred_actions = policy.apply_fn(
			{"params": params},
			observations=observations,
			skills=skills,
			rngs={"dropout": dropout_key},
			deterministic=False,
			training=True
		)
		pred_actions = pred_actions.reshape(-1, action_dim) * maskings
		mse_loss = jnp.sum(jnp.mean((pred_actions - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)

		_infos = {"policy/mse_loss": mse_loss}
		return mse_loss, _infos

	new_mlp, infos = policy.apply_gradient(loss_fn)
	return new_mlp, infos


@jax.jit
def retail_transfer_update(
	rng: jnp.ndarray,
	policy: Model,
	transfer: Model,
	source_observations: jnp.ndarray,
	flat_source_skills: jnp.ndarray,
	flat_target_skills: jnp.ndarray,
	target_skill_sequence: jnp.ndarray,	# [b, M, d]
	language: jnp.ndarray,	# [b, d]
	obs_maskings: jnp.ndarray,	# [b, l] (Source obs); Therefore same with act_masking.
	act_maskings: jnp.ndarray, 	# [b, l] (Source act); Therefore same with obs_masking.
):
	"""
	이게 시발 될 리가 없는데 ...
		Input of transfer: [target observation, source action, language]
		Want to predict: [target action]
	"""
	rng, dropout_key = jax.random.split(rng)

	a_tgt = policy.apply_fn(
		{"params": policy.params},
		observations=source_observations,
		skills=flat_target_skills,
		rngs={"dropout": dropout_key},
		deterministic=False,
		training=True
	)
	a_tgt = a_tgt[:, -1, ...]

	a_src = policy.apply_fn(
		{"params": policy.params},
		observations=source_observations,
		skills=flat_source_skills,
		rngs={"dropout": dropout_key},
		deterministic=False,
		training=True
	)
	a_src = a_src[:, [-1], ...]

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		pred_target_actions = transfer.apply_fn(
			{"params": params},
			observations=source_observations,
			source_actions=a_src,
			target_skill_sequence=target_skill_sequence,
			language=language,
			obs_maskings=obs_maskings,
			act_maskings=act_maskings[:, [-1]],
			rngs={"dropout": dropout_key},
			deterministic=False,
		)

		transfer_loss = jnp.sum(jnp.mean((pred_target_actions - a_tgt) ** 2, axis=-1)) / jnp.sum(act_maskings[:, -1])
		_infos = {"transfer/mse_loss": transfer_loss, "__pred": pred_target_actions, "__target": a_tgt}
		return transfer_loss, _infos

	new_transfer, infos = transfer.apply_gradient(loss_fn)
	return new_transfer, infos
