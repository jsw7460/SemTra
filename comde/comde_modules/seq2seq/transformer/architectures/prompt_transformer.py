from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.idx_decoder import TransformerIndexInputDecoder
from comde.comde_modules.seq2seq.transformer.architectures.encoder import TransformerEncoder



LOGITS_MAX = 15.0
LOGITS_MIN = -4.0


class PrimPromptLearningTransformer(nn.Module):
	# This is language model. Output a natural language (or its index)
	encoder_cfg: Dict
	decoder_cfg: Dict
	input_dropout_prob: float
	vocab_size: int

	input_dropout = None
	input_layer = None
	encoder = None
	decoder = None

	lm_head = None

	def setup(self) -> None:
		"""
		Inputs:
			x - Input features of shape [Batch, SeqLen, input_dim]
			mask - Mask to apply on the attention outputs (optional)
			add_positional_encoding - If True, we add the positional encoding to the input.
									  Might not be desired for some tasks.
			train - If True, dropout is stochastic
		"""

		self.input_dropout = nn.Dropout(self.input_dropout_prob)
		self.encoder = TransformerEncoder(**self.encoder_cfg)
		self.decoder = TransformerIndexInputDecoder(**self.decoder_cfg, vocab_size=self.vocab_size)

		self.lm_head = create_mlp(
			output_dim=self.vocab_size,
			net_arch=[],
			layer_norm=True,
			squash_output=False
		)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		x: jnp.ndarray,  # [b, l]	# Index ! Not a word embedding
		encoder_q: jnp.ndarray,	# This is word embedding from pretrained BERT
		encoder_kv: jnp.ndarray,	# This is word embedding from pretraiend BERT
		q_mask: jnp.ndarray,
		kv_mask: jnp.ndarray,
		deterministic: bool = False,
		*args, **kwargs  # Do not remove this
	) -> Dict[str, jnp.ndarray]:

		context, encoder_attention_weights = self.encoder(q=encoder_q, kv=encoder_kv, q_mask=q_mask, kv_mask=kv_mask, deterministic=deterministic)
		decoded_x, decoder_attention_weights = self.decoder(x, context, q_mask, deterministic=deterministic)

		pred = self.lm_head(decoded_x)  # [b, l, vocab_size]
		info = {
			"pred": pred,
			"encoder_attention_weights": encoder_attention_weights,
			"decoder_attention_weights": decoder_attention_weights
		}

		return info
