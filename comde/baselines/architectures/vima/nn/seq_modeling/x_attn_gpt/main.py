from typing import Any, Dict, Optional, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.scope import FrozenVariableDict

from comde.baselines.architectures.vima.nn.seq_modeling.x_attn_gpt.components import (
	Block, XAttention)

from comde.comde_modules.seq2seq.transformer.architectures.decoder_block import DecoderBlock


class XAttnGPT(nn.Module):
	embed_dim: int = 768
	num_positions: int = 512
	num_layers: int = 12
	num_heads: int = 12
	use_bias: bool = True
	layer_norm_epsilon: float = 1e-5
	attn_scale: bool = False
	dropout_rate: float = 0.1
	activation: str = "relu"
	xattn_num_heads: int = 8
	xattn_ff_dim: int = 4
	xattn_detach_qk: bool = False
	xattn_num_positions: int = 512
	use_geglu: bool = False

	dropout = None
	xattns = None

	def setup(self) -> None:
		self.dropout = nn.Dropout(rate=self.dropout_rate)
		self.xattns = [
			DecoderBlock(
				input_dim=self.embed_dim,
				num_heads=self.xattn_num_heads,
				ff_dim=self.xattn_ff_dim,
				dropout_prob=self.dropout_rate,
				activation_fn=nn.relu,
				use_bias=True,
			)
		]

	def __call__(
		self,
		*,
		obs_action_tokens: jnp.ndarray,
		obs_action_position_ids: Optional[jnp.ndarray] = None,
		prompt_tokens: jnp.ndarray,
		prompt_mask: Optional[jnp.ndarray] = None,
		prompt_position_ids: Optional[jnp.ndarray] = None,
		batch_first: bool = False,
		obs_action_masks: Optional[jnp.ndarray] = None,
		deterministic: bool,
	):
		b, l, e = obs_action_tokens.shape
		input_shape = obs_action_tokens.shape[:-1]

		output_shape = input_shape + (obs_action_tokens.shape[-1],)
		# assert prompt_tokens.shape[1] <= self.xattn_position_ids.shape[0]

		for xattn in self.xattns:
			obs_action_tokens, _ = xattn(
				q=obs_action_tokens,
				kv=prompt_tokens,
				mask=prompt_mask,
				deterministic=deterministic
			)

		obs_action_tokens = obs_action_tokens.reshape(output_shape)
		assert obs_action_tokens.shape == (b, l, e)
		if not batch_first:
			obs_action_tokens = obs_action_tokens.transpose(0, 1)

		return obs_action_tokens

	def _check_input(
		self,
		obs_action_tokens: jnp.ndarray,
		prompt_tokens: jnp.ndarray,
		prompt_mask: Optional[jnp.ndarray] = None,
		batch_first: bool = False,
		obs_action_masks: Optional[jnp.ndarray] = None,
	):
		assert obs_action_tokens.ndim == 3
		assert obs_action_tokens.dtype == jnp.float32
		assert prompt_tokens.ndim == 3
		assert prompt_tokens.dtype == jnp.float32

		if batch_first:
			B_oa, L_oa, E_oa = obs_action_tokens.shape
			B_p, L_p, E_p = prompt_tokens.shape
		else:
			L_oa, B_oa, E_oa = obs_action_tokens.shape
			L_p, B_p, E_p = prompt_tokens.shape
		assert B_oa == B_p
		assert E_oa == E_p
		B = B_oa

		if prompt_mask is not None:
			# fmt: off
			assert prompt_mask.shape == (B, L_p) or prompt_mask.shape == (B, 1, L_p), \
				f"Expect `prompt_mask` to have shape of either ({B, 1, L_p}) or ({B, L_p}), but got {prompt_mask.shape}"
			# fmt: on
			# a simple sanity check on the mask
			assert jnp.all(
				prompt_mask.sum(axis=-1) > 0
			), "each source token should attend to at least one target token"
			assert prompt_mask.dtype == jnp.bool_
		if obs_action_masks is not None:
			assert obs_action_masks.shape == (B, L_oa)
			assert jnp.all(obs_action_masks.sum(axis=-1) > 0)
			assert obs_action_masks.dtype == jnp.bool_
