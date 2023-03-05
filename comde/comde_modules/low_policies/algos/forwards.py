import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model


@jax.jit
def decisiontransformer_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	rtgs: jnp.ndarray
):
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
	rtgs: jnp.ndarray
):
	rng, _ = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		actions=actions,
		skills=skills,
		timesteps=timesteps,
		maskings=maskings,
		rtgs=rtgs,
		deterministic=True
	)

	return rng, prediction