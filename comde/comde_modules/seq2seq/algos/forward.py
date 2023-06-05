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
	encoder_q: jnp.ndarray,
	encoder_kv: jnp.ndarray,
	q_mask: jnp.ndarray,
	kv_mask: jnp.ndarray,
	deterministic: bool = True
) -> Tuple[PRNGKey, Dict[str, jnp.ndarray]]:
	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		x=x,
		encoder_q=encoder_q,
		encoder_kv=encoder_kv,
		q_mask=q_mask,
		kv_mask=kv_mask,
		deterministic=deterministic,
		rngs={"dropout": dropout_key}
	)
	# prediction is of the form {"pred_skills": ..., "encoder_attention_weights": ..., ...}.
	return rng, prediction