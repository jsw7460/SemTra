from typing import Any, Dict, Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def vima_update(
	rng: jnp.ndarray,
	policy: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,  # [b, l, d]
	timesteps: jnp.ndarray,
	prompts: jnp.ndarray,  # [b, M]
	prompt_assets: jnp.ndarray, # [b, N, d]
	prompts_maskings: jnp.ndarray,  # [b, M]: indicate whether dummy source skills
	prompt_assets_maskings: jnp.ndarray, # [b, N]: indicate whether dummy source skills
	maskings: jnp.ndarray,  # [b, l]
) -> Tuple[Model, Dict[str, Any]]:
	rng, dropout_key, dist_key = jax.random.split(rng, 3)
	action_dim = actions.shape[-1]

	target_actions = actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, Any]]:
		action_preds = policy.apply_fn(
			{"params": params},
			observations=observations,
			observations_mask=maskings,
			actions=actions,
			timesteps=timesteps,
			prompt=prompts,
			prompt_assets=prompt_assets,
			prompt_mask=prompts_maskings,
			prompt_assets_mask=prompt_assets_maskings,
            deterministic=False,
			rngs={"dropout": dropout_key, "dist": dist_key}
		)

		action_preds = action_preds.reshape(-1, action_dim) * maskings.reshape(-1, 1)
		action_loss = jnp.sum(jnp.mean((action_preds - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)
		_infos = {"promptdt/bc_loss": action_loss, "action_preds": action_preds, "target_actions": target_actions}
		return action_loss, _infos

	new_policy, info = policy.apply_gradient(loss_fn)
	return new_policy, info


@jax.jit
def flaxvima_update(
	rng: jnp.ndarray,
	policy: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,  # [b, l, d]
	maskings: jnp.ndarray,  # [b, l]
	timesteps: jnp.ndarray,
	params_for_skills: jnp.ndarray,
	prompts: Dict[str, jnp.ndarray],  # [b, M]
	prompts_maskings: Dict[str, jnp.ndarray],  # [b, M]: indicate whether dummy source skills
) -> Tuple[Model, Dict[str, Any]]:
	rng, dropout_key, dist_key = jax.random.split(rng, 3)
	action_dim = actions.shape[-1]

	target_actions = actions.reshape(-1, action_dim) * maskings.reshape(-1, 1)

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, Any]]:
		action_preds = policy.apply_fn(
			{"params": params},
			observations=observations,
			maskings=maskings,
			actions=actions,
			timesteps=timesteps,
			param_for_skills=params_for_skills,
            deterministic=False,
			**prompts, **prompts_maskings,
			rngs={"dropout": dropout_key, "dist": dist_key}
		)

		action_preds = action_preds.reshape(-1, action_dim) * maskings.reshape(-1, 1)
		action_loss = jnp.sum(jnp.mean((action_preds - target_actions) ** 2, axis=-1)) / jnp.sum(maskings)
		_infos = {"promptdt/bc_loss": action_loss, "action_preds": action_preds, "target_actions": target_actions}
		return action_loss, _infos

	new_policy, info = policy.apply_gradient(loss_fn)
	return new_policy, info