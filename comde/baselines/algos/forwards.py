from typing import Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@jax.jit
def flatbc_forward(
	rng: jnp.ndarray,
	model: Model,
	policy_input: jnp.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, _ = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		x=policy_input,
		deterministic=True
	)
	return rng, prediction


@jax.jit
def demogen_gravity_forward(
	rng: jnp.ndarray,
	model: Model,
	model_input: jnp.ndarray
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, _ = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		x=model_input,
		deterministic=True
	)
	return rng, prediction


demogen_policy_forward = flatbc_forward


@jax.jit
def promptdt_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	rtgs: jnp.ndarray,
	prompts: jnp.ndarray,
	prompts_maskings: jnp.ndarray,
	sequential_requirement: jnp.ndarray,
	non_functionality: jnp.ndarray,
	param_for_skills: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray
):
	rng, _ = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
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
		deterministic=True
	)
	return rng, prediction


@jax.jit
def vima_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,  # d_o
	observations_mask: jnp.ndarray,
	actions: jnp.ndarray,  # d_a
	timesteps: jnp.ndarray,
	prompt: jnp.ndarray,
	prompt_assets: jnp.ndarray,
	prompt_mask: jnp.ndarray,
	prompt_assets_mask: jnp.ndarray,
):
	rng, _ = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		observations_mask=observations_mask,
		actions=actions,
		timesteps=timesteps,
		prompt=prompt,
		prompt_assets=prompt_assets,
		prompt_mask=prompt_mask,
		prompt_assets_mask=prompt_assets_mask,
		deterministic=True
	)
	return rng, prediction
