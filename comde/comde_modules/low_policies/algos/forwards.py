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
	deterministic: bool = True
):
	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		actions=actions,
		skills=skills,
		timesteps=timesteps,
		maskings=maskings,
		deterministic=deterministic,
		rngs={"dropout": dropout_key}
	)

	return rng, prediction


@jax.jit
def skill_mlp_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	timesteps: jnp.ndarray,
	maskings: jnp.ndarray,
	deterministic: bool = True
):
	# fixed: originally obs-skill concatenation was done here, 
	# 	     moved to forward function in PrimSkillMLP
	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		actions=actions,
		skills=skills,
		timesteps=timesteps,
		maskings=maskings,
		# model_input,
		rngs={"dropout": dropout_key},
		deterministic=deterministic,
		training=False
	)
	_,act_pred,_ = prediction
	return rng, act_pred

