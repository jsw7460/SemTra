from typing import Dict, Tuple

import jax
from jax import numpy as jnp

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


@jax.jit
def skilltoskill_transformer_ce_updt(
	rng: jnp.ndarray,
	tr: Model,  # transformer
	context: jnp.ndarray,  # [b, X, d] : Cancatenation of source skills and language operator (calculated outside)
	context_mask: jnp.ndarray,  # [b, X, d]
	target_skills: jnp.ndarray,
	target_skills_idxs: jnp.ndarray,
	observations: jnp.ndarray,
	actions: jnp.ndarray,
	skills: jnp.ndarray,
	maskings: jnp.ndarray,
	n_source_skills: jnp.ndarray,  # [b, ]
	n_target_skills: jnp.ndarray,  # [b, ]
	start_token: jnp.ndarray,  # [d, ]
) -> Tuple[Model, Dict]:
	rng, dropout_key = jax.random.split(rng)
	# Note: Target skill을 한 칸 밀어줘야 함 (Due to start token)

	skill_dim = skills.shape[-1]
	batch_size = n_source_skills.shape[0]
	context_maxlen = context.shape[1]

	start_token = jnp.broadcast_to(start_token, shape=(batch_size, 1, skill_dim))

	input_skills = jnp.concatenate((start_token, target_skills), axis=1)

	target_skills_idxs = jnp.concatenate((target_skills_idxs, -1 + jnp.zeros((batch_size, 1), dtype="i4")), axis=-1)
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
			context=context,
			mask=context_mask,
			rngs={"dropout": dropout_key}
		)
		pred_skills = model_output["pred_skills"]  # [b, M, n_skills]

		# Learn target skills.
		pred_skills = jax.nn.softmax(pred_skills, axis=-1)  # [b, M, n_skills] (Probability vec)
		tgt_skills = target_skills_idxs[..., jnp.newaxis]  # [b, M, 1]

		likelihood = jnp.log(pred_skills)  # [b, M, n_skills]
		likelihood = jnp.take_along_axis(likelihood, tgt_skills, axis=2)  # [b, M, 1]
		likelihood = jnp.squeeze(likelihood, axis=-1)

		tgt_mask = jnp.arange(target_max)
		tgt_mask = jnp.broadcast_to(tgt_mask, (batch_size, target_max))  # [b, M]
		tgt_mask = jnp.where(tgt_mask < n_target_skills[..., jnp.newaxis], 1, 0)  # [b, M]

		# ce_loss = - likelihood
		ce_loss = - likelihood * tgt_mask
		skills_loss = jnp.sum(ce_loss) / jnp.sum(tgt_mask)

		loss = skills_loss

		pred_skills_idxs = jnp.argmax(pred_skills, axis=-1)  # [b, M]
		pred_skills_idxs = jnp.where(tgt_mask == 1, pred_skills_idxs, -1)
		target_skills_answer = jnp.where(tgt_mask == 1, target_skills_idxs, -2)

		match_ratio = jnp.sum(pred_skills_idxs == target_skills_answer) / jnp.sum(tgt_mask)

		_info = {
			"s2s/skill_loss(ce)": skills_loss,
			"s2s/match_ratio": match_ratio,
			"__tgt_mask": tgt_mask,
			"__pred_skills": jnp.log(pred_skills),
			"__tgt_skills": tgt_skills,
			"__s2s/obs": observations,
			"__s2s/act": actions,
			"__s2s/maskings": maskings,
			"__s2s/context_maxlen": context_maxlen,
			"__skills": skills,
			"__likelihood": likelihood,
			"__pred_skills_idxs": pred_skills_idxs,
			"__target_skills_answer": target_skills_answer
		}
		return loss, _info

	new_tr, info = tr.apply_gradient(loss_fn=loss_fn)

	return new_tr, info
