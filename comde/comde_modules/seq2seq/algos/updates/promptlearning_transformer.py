from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def promptlearning_transformer_updt(
	rng: jnp.ndarray,
	tr: Model,
	encoder_q: jnp.ndarray,  # [b, l, d]: Pretrained word embedding
	encoder_kv: jnp.ndarray,  # [b, l, d]: Pretrained word embedding
	q_mask: jnp.ndarray,
	kv_mask: jnp.ndarray,
	decoder_idxs: jnp.ndarray,  # [b, l]: Not a word embedding
	decoder_masks: jnp.ndarray,  # [b, l]
):
	rng, dropout_key = jax.random.split(rng)

	decoder_input = decoder_idxs[:, :-1]	# Remove EOS token
	decoder_target = decoder_idxs[:, 1:]	# Remove BOS token
	decoder_masks = decoder_masks[:, 1:]	# Remove BOS token

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
		model_output = tr.apply_fn(
			{"params": params},
			x=decoder_input,
			encoder_q=encoder_q,
			encoder_kv=encoder_kv,
			q_mask=q_mask,
			kv_mask=kv_mask,
			rngs={"dropout": dropout_key}
		)
		prediction = model_output["pred"]  # [b, l, vocab_size]

		logits = jnp.take_along_axis(prediction, decoder_target[..., jnp.newaxis], axis=-1)  # [b, l, 1]
		logits = jnp.squeeze(logits)		# [b, l]
		logits = logits * decoder_masks

		loss = - jnp.mean(logits)

		_info = {
			"__model_output": model_output,
			"__prediction": prediction,
			"prompt_tr/loss(nl)": loss
		}

		return loss, _info

	new_tr, info = tr.apply_gradient(loss_fn)

	return new_tr, info
