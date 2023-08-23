from functools import partial
from typing import Tuple, Dict

import jax
from jax import numpy as jnp

from comde.comde_modules.environment_encoder.architectures.vq_lstm import PrimVecQuantizedLSTM
from comde.utils.jax_utils.general import jnp_polyak_update

from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params

INFTY = 1E+8
EPSILON = 1E-8


@jax.jit
def vqlstm_contrastive_update(
	rng: jnp.ndarray,
	encoder: Model,
	first_history_obs: jnp.ndarray,
	second_history_obs: jnp.ndarray,
	first_history_act: jnp.ndarray,
	second_history_act: jnp.ndarray,
	first_lstm_n_iter: jnp.ndarray,
	second_lstm_n_iter: jnp.ndarray,
	coef_positive_loss: float,
	coef_negative_loss: float
):
	rng, dropout_key, init_carry = jax.random.split(rng, 3)
	first_history = jnp.concatenate((first_history_obs, first_history_act), axis=2)
	second_history = jnp.concatenate((second_history_obs, second_history_act), axis=2)

	batch_size = first_history.shape[0]
	self_mask = jnp.eye(2 * batch_size, dtype="i4")
	pos_mask = jnp.roll(self_mask, shift=batch_size, axis=0)

	def encoder_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		# First augmentation
		first_batch_info = encoder.apply_fn(
			{"params": params},
			sequence=first_history,
			n_iter=first_lstm_n_iter,
			deterministic=False,
			rngs={"dropout": dropout_key, "init_carry": init_carry, "sampling": rng}
		)
		first_unquantized_h = first_batch_info["unquantized"]
		first_quantized_h = first_batch_info["quantized"]
		first_hidden_vars = first_unquantized_h + jax.lax.stop_gradient(first_quantized_h - first_unquantized_h)

		second_batch_info = encoder.apply_fn(
			{"params": params},
			sequence=second_history,
			n_iter=second_lstm_n_iter,
			deterministic=False,
			rngs={"dropout": dropout_key, "init_carry": init_carry, "sampling":rng}
		)
		second_unquantized_h = second_batch_info["unquantized"]
		second_quantized_h = second_batch_info["quantized"]
		second_hidden_vars = second_unquantized_h + jax.lax.stop_gradient(second_quantized_h - second_unquantized_h)

		features = jnp.concatenate((first_hidden_vars, second_hidden_vars), axis=0)
		normalized_features = features / jnp.linalg.norm(features, axis=-1, keepdims=True)

		similarity = jnp.einsum("ij, kj -> ik", normalized_features, normalized_features)
		similarity = jnp.where(self_mask > 0, -INFTY, similarity)

		# positive_loss = - coef_positive_loss * jnp.mean(similarity[pos_mask==1])
		positive_loss = - coef_positive_loss * jnp.sum(jnp.where(pos_mask==1, similarity, 0.0))
		negative_loss = coef_negative_loss * jnp.mean(jax.nn.logsumexp(similarity, axis=-1, keepdims=True))
		contrastive_loss = positive_loss + negative_loss
		_info = {
			"dyna_encoder/contrastive_loss": contrastive_loss,
			"dyna_encoder/positive_loss": positive_loss,
			"dyna_encoder/negative_loss": negative_loss,
			"__similarity": similarity,
			"__pos_mask": pos_mask
		}

		return contrastive_loss, _info

	new_encoder, info = encoder.apply_gradient(encoder_loss_fn)
	return new_encoder, info


@partial(jax.jit, static_argnames=("coef_skill_dec_aid", "n_codebook", "subseq_len", "coef_dyna_decoder"))
def vq_lstm_update(
	rng: jnp.ndarray,
	task_encoder: Model,
	task_decoder: Model,
	low_policy: Model,
	observations: jnp.ndarray,	# [History + current]
	actions: jnp.ndarray,	# [History + current]
	skills: jnp.ndarray,	# [History + current]
	lstm_n_iter: jnp.ndarray,  # [batch_size,]
	n_codebook: int,
	coef_skill_dec_aid: float,
	coef_commitment: float,  # beta in VQ-vae paper
	coef_gamma_moving_avg: float,
	coef_dyna_decoder: float
):
	"""
	Same with above function except that lower policy is transformer.

	# TODO: In original paper, the prior (over discrete latent) is uniform distribution.
	        So we don't need to use prior regularization (Its just constant !)
	        Or, regularizing the unquantized_h to the normal distribution makes sense, but it's effict is mysterious.
		In this function, we calculate the following losses:
			1. Reconstruction
			2. Aid skill decoder
			3. Move codebook to center of embeddings 'by means of exponentially moving average' (not gradient flow)
			4. Commitment loss
		Note: 1 and 2 uses the straight-through gradient estimator.
		Note: 3 is updated outside the nested loss function, since it has nothing to do with 'gradients' of the encoder.
	"""
	rng, dropout_key, encoder_key, hidden_key = jax.random.split(rng, 4)
	subseq_len = observations.shape[1] - 1  # Due to current, we subtract by 1

	history_observations = observations[:, :-1, ...]  # use only history
	history_next_observations = observations[:, 1:, ...]  # use until current
	history_actions = actions[:, :-1, ...]  # use only history

	current_actions = actions[:, -1, ...]
	encoder_input = jnp.concatenate((history_observations, history_actions), axis=2)

	def encoder_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		# 1. Reconstruction
		encoder_info = task_encoder.apply_fn(
			{"params": params},
			sequence=encoder_input,
			n_iter=lstm_n_iter,
			deterministic=False,
			rngs={"dropout": dropout_key, "init_carry": encoder_key},
		)
		_codebook_idxs = encoder_info["nearest_codebook_idxs"]
		_quantized_h = encoder_info["quantized"]
		_unquantized_h = encoder_info["unquantized"]

		# Used for straight-through gradient estimator!
		# Same with quantized_h, but gradient is flowed by unquantized_h (Since quantized and unquantized are similar !)
		auxiliary_input = _unquantized_h + jax.lax.stop_gradient(_quantized_h - _unquantized_h)

		repeated_hidden_variable = jnp.repeat(auxiliary_input[:, jnp.newaxis, ...], repeats=subseq_len, axis=1)
		__decoder_input = jnp.concatenate(
			arrays=(repeated_hidden_variable, history_observations, history_next_observations),
			axis=-1
		)
		decoded_transitions = task_decoder.apply_fn(
			{"params": task_decoder.params},
			__decoder_input,
			deterministic=False,
			training=True
		)
		reconstruction_loss = jnp.mean((decoded_transitions - history_actions) ** 2)

		# 2. Aid skill decoder
		if coef_skill_dec_aid > 0:
			# Since the parameter of encoder should be changed by the skill decoder, we use auxiliary input
			context_skills = jnp.concatenate((skills[:, -1, ...], _quantized_h), axis=-1)
			predicted_actions = low_policy.apply_fn(
				{"params": low_policy.params},
				observations[:, -1, ...],
				context_skills,
				rngs={"dropout": dropout_key},
				deterministic=False,
				training=True
			)
			skill_decoder_aid_loss = coef_skill_dec_aid * jnp.mean((current_actions - predicted_actions) ** 2)
		else:
			skill_decoder_aid_loss = 0.0

		# 4. Commitment loss
		current_embedding = task_encoder.apply_fn(
			{"params": task_encoder.params},  # No gradient for here
			rngs={"dropout": dropout_key},
			method=PrimVecQuantizedLSTM.get_current_codebook
		)
		current_embedding = jax.lax.stop_gradient(current_embedding)
		commitment_loss = 0.0
		for k in range(n_codebook):
			kth_idx = jnp.where(_codebook_idxs == k, 1, 0)
			kth_mean = jnp.sum(_unquantized_h * kth_idx.reshape(-1, 1), axis=0) / (jnp.sum(kth_idx) + EPSILON)
			commitment_loss += jnp.mean((kth_mean - current_embedding.at[k].get()) ** 2)

		commitment_loss = coef_commitment * commitment_loss
		task_encoder_loss = reconstruction_loss + skill_decoder_aid_loss + commitment_loss

		# Before return, compute input of task decoder to update it. (This use quantized h, not auxiliary input)
		_repeated_quantized_h = jnp.repeat(_quantized_h[:, jnp.newaxis, ...], repeats=subseq_len, axis=1)
		_unquantized_decoder_input = jnp.concatenate(
			arrays=(_repeated_quantized_h, history_observations, history_next_observations),
			axis=-1
		)
		_infos = {
			"dyna_encoder/loss": task_encoder_loss,
			"dyna_encoder/reconstruction_loss": reconstruction_loss,
			"dyna_encoder/skill_dec_aid_loss": skill_decoder_aid_loss,
			"dyna_encoder/commitment_loss": commitment_loss,
			"__unquantized_h": _unquantized_h,
			"__quantized_h": _quantized_h,
			"__decoder_input": _unquantized_decoder_input,
			"__codebook_idxs": _codebook_idxs
		}
		return task_encoder_loss, _infos

	middle_task_encoder, info = task_encoder.apply_gradient(encoder_loss_fn)

	codebook_idxs = info["__codebook_idxs"]
	unquantized_h = info["__unquantized_h"]

	# 3. Move codebook to center of embeddings 'by means of exponentially moving average'
	codebook_dim = unquantized_h.shape[-1]
	moving_average_target = jnp.empty((0, codebook_dim))

	for i in range(n_codebook):
		ith_idx = jnp.where(codebook_idxs == i, 1, 0)
		ith_mean = jnp.sum(unquantized_h * ith_idx.reshape(-1, 1), axis=0) / (jnp.sum(ith_idx + EPSILON))
		moving_average_target = jnp.vstack((moving_average_target, ith_mean))

	updated_codebook = middle_task_encoder.apply_fn(
		{"params": middle_task_encoder.params},
		method=PrimVecQuantizedLSTM.get_current_codebook
	)
	new_codebook = jnp_polyak_update(updated_codebook, moving_average_target, tau=coef_gamma_moving_avg)
	new_codebook = jax.tree_map(
		lambda p, tp: p * coef_gamma_moving_avg + tp * (1 - coef_gamma_moving_avg),
		middle_task_encoder.params["codebook"]["embedding"],
		new_codebook
	)
	new_params = middle_task_encoder.params
	new_params["codebook"]["embedding"] = new_codebook
	new_encoder = middle_task_encoder.replace(params=new_params)

	decoder_input = info["__decoder_input"]
	decoder_input = jax.lax.stop_gradient(decoder_input)

	# Done: Task encoder

	def decoder_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
		decoded_transitions = task_decoder.apply_fn(
			{"params": params},
			decoder_input,
			rngs={"dropout": dropout_key},
			deterministic=False,
			training=True
		)
		decoder_loss = coef_dyna_decoder * jnp.mean((decoded_transitions - history_actions) ** 2)  # Reconstruction loss
		return decoder_loss, {"task_decoder/loss": decoder_loss}

	new_decoder, decoder_info = task_decoder.apply_gradient(decoder_loss_fn)

	return new_encoder, new_decoder, {**info, **decoder_info}