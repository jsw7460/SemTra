from typing import Tuple, Dict
from functools import partial

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import PRNGKey


@partial(jax.jit, static_argnames=("batch_size", ))
def skilltoskill_model_forward(
	rng: jnp.ndarray,
	model: Model,
	sequence: jnp.ndarray,
	batch_size: int,
) -> Tuple[PRNGKey, jnp.ndarray]:
	rng, dropout_key, carry_key = jax.random.split(rng, 3)
	stacked_output = model.apply_fn(
		{"params": model.params},
		sequence,
		batch_size,
		rngs={"dropout": dropout_key, "init_carry": carry_key},
	)
	return rng, stacked_output


@jax.jit
def skilltoskill_transformer_forward(
	rng: jnp.ndarray,
	model: Model,
	x: jnp.ndarray,
	context: jnp.ndarray,	# [b, L, d]
	mask: jnp.ndarray,	# Padding mask for the context. Not a causal mask.
	deterministic: bool = True
) -> Tuple[PRNGKey, Dict[str, jnp.ndarray]]:
	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		x=x,
		context=context,
		mask=mask,
		deterministic=deterministic,
		rngs={"dropout": dropout_key}
	)
	# prediction is of the form {"pred_skills": ..., "pred_intent": ...}.
	return rng, prediction