from typing import Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@jax.jit
def decisiontransformer_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	rtgs: jnp.ndarray
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, _ = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		actions=actions,
		timesteps=timesteps,
		maskings=maskings,
		rtgs=rtgs,
		deterministic=True
	)

	return rng, prediction


@jax.jit
def skill_decisiontransformer_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	deterministic: bool = True
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, dropout_key = jax.random.split(rng)
	action_preds = model.apply_fn(
		{"params": model.params},
		observations=observations,
		actions=actions,
		skills=skills,
		timesteps=timesteps,
		maskings=maskings,
		deterministic=deterministic,
		rngs={"dropout": dropout_key}
	)

	return rng, action_preds


@jax.jit
def skill_mlp_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	skills: jnp.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:

	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		skills=skills,
		rngs={"dropout": dropout_key},
		deterministic=True,
		training=False
	)
	return rng, prediction



@jax.jit
def skill_ln_mlp_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	skills: jnp.ndarray,
	non_functionality: jnp.ndarray,
	parameters: jnp.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:

	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		skills=skills,
		non_functionality=non_functionality,
		parameters=parameters,
		rngs={"dropout": dropout_key},
		deterministic=True,
		training=False
	)
	return rng, prediction


@jax.jit
def skill_promptdt_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	prompts: jnp.ndarray,
	prompts_maskings: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	deterministic: bool = True
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, dropout_key = jax.random.split(rng)
	action_preds = model.apply_fn(
		{"params": model.params},
		observations=observations,
		actions=actions,
		skills=skills,
		prompts=prompts,
		prompts_maskings=prompts_maskings,
		timesteps=timesteps,
		maskings=maskings,
		deterministic=deterministic,
		rngs={"dropout": dropout_key}
	)

	return rng, action_preds