from typing import Dict, Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params
from functools import partial


@partial(jax.jit, static_argnames=("eos_token_idx", ))
def skilltoskill_transformer_ce_updt(
	rng: jnp.ndarray,
	tr: Model,  # transformer
	encoder_q: jnp.ndarray,  # [b, X, d] : Cancatenation of source skills and language operator (calculated outside)
	encoder_kv: jnp.ndarray,
	q_mask: jnp.ndarray,
	kv_mask: jnp.ndarray,
	target_skills: jnp.ndarray,  # Language embedding. Not an index (Input of transformer decoder)
	target_skills_idxs: jnp.ndarray,  # Index. (Target of transformer)
	n_source_skills: jnp.ndarray,  # [b, ]
	n_target_skills: jnp.ndarray,  # [b, ]
	bos_token: jnp.ndarray,  # [d, ]
	eos_token_idx: int,
) -> Tuple[Model, Dict]:
	rng, dropout_key = jax.random.split(rng)
	# Note: Target skill을 한 칸 밀어줘야 함 (Due to start token)

	token_dim = bos_token.shape[-1]
	batch_size = n_source_skills.shape[0]

	bos_token = jnp.broadcast_to(bos_token, shape=(batch_size, 1, token_dim))
	input_skills = jnp.concatenate((bos_token, target_skills), axis=1)

	# Concatenate with end token
	target_skills_idxs = jnp.concatenate(
		(target_skills_idxs, eos_token_idx + jnp.zeros((batch_size, 1), dtype="i4")),
		axis=-1
	)
	target_max = target_skills_idxs.shape[1]

	def loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
		"""
		ctx_padding_mask:
			This is padding mask for the context, not a causal mask.
			Mask for the answer is not required since the loss function is not calculated for that words.
		"""

		model_output = tr.apply_fn(
			{"params": params},
			x=input_skills,
			encoder_q=encoder_q,
			encoder_kv=encoder_kv,
			q_mask=q_mask,
			kv_mask=kv_mask,
			rngs={"dropout": dropout_key}
		)
		pred_skills = model_output["pred_logits"]  # [b, M, n_skills]
		pred_skills_prob = pred_skills
		# Learn target skills.
		pred_skills = jax.nn.softmax(pred_skills, axis=-1)  # [b, M, n_skills] (Probability vec)
		tgt_skills = target_skills_idxs[..., jnp.newaxis]  # [b, M, 1]

		likelihood = jnp.log(pred_skills)  # [b, M, n_skills]
		likelihood = jnp.take_along_axis(likelihood, tgt_skills, axis=2)  # [b, M, 1]
		likelihood = jnp.squeeze(likelihood, axis=-1)

		tgt_mask = jnp.arange(target_max)
		tgt_mask = jnp.broadcast_to(tgt_mask, (batch_size, target_max))  # [b, M]
		# Do not include equality here for evaluation
		tgt_mask_wo_last = jnp.where(tgt_mask < n_target_skills[..., jnp.newaxis], 1, 0)
		# Include equal since we predict end token
		tgt_mask = jnp.where(tgt_mask <= n_target_skills[..., jnp.newaxis], 1, 0)  # [b, M]

		ce_loss = - likelihood * tgt_mask
		skills_loss = jnp.sum(ce_loss) / jnp.sum(tgt_mask)

		loss = skills_loss

		prompt_dict = {}
		target_skills_mask = None
		attention_weights = None

		pred_skills_idxs = jnp.argmax(pred_skills, axis=-1)  # [b, M]
		_pred_skills_wo_mask = pred_skills_idxs
		pred_skills_idxs = jnp.where(tgt_mask_wo_last == 1, pred_skills_idxs, -1)
		target_skills_answer = jnp.where(tgt_mask_wo_last == 1, target_skills_idxs, -2)

		match_ratio = jnp.sum(pred_skills_idxs == target_skills_answer) / jnp.sum(tgt_mask_wo_last)

		_info = {
			"s2s/skill_loss(ce)": skills_loss,
			"s2s/match_ratio(%)": match_ratio * 100,
			"__model_output": model_output,
			"__tgt_mask": tgt_mask,
			"__tgt_mask_wo_last": tgt_mask_wo_last,
			"__pred_skills": jnp.log(pred_skills),
			"__pred_skills_prob": pred_skills_prob,
			"__tgt_skills": tgt_skills,
			"__likelihood": likelihood,
			"__pred_skills_idxs": pred_skills_idxs,
			"__pred_skills_wo_mask": _pred_skills_wo_mask,
			"__target_skills_answer": target_skills_answer,
			"__input_skills": input_skills,
			"__encoder_attention_weights": model_output["encoder_attention_weights"],
			"__decoder_attention_weights": model_output["decoder_attention_weights"],
			"__prompt_dict": prompt_dict,
			"__target_skills_mask": target_skills_mask,
			"__attention_weights": attention_weights,
			"__q_mask": q_mask
		}

		return loss, _info

	new_tr, info = tr.apply_gradient(loss_fn=loss_fn)

	return new_tr, info
