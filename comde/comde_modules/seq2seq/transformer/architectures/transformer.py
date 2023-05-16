from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.seq2seq.transformer.architectures.decoder import TransformerDecoder
from comde.comde_modules.seq2seq.transformer.architectures.encoder import TransformerEncoder


class PrimSklToSklIntTransformer(nn.Module):
	# Output: Skill sequence
	encoder_cfg: Dict
	decoder_cfg: Dict
	input_dropout_prob: float
	skill_dropout_prob: float
	skill_pred_dim: int
	non_functionality_dim: int
	param_dim: int

	input_dropout = None
	input_layer = None
	encoder = None
	decoder = None

	pred_skills = None
	pred_nonfunc = None
	pred_param = None

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
		self.decoder = TransformerDecoder(**self.decoder_cfg)

		self.pred_skills = create_mlp(
			output_dim=self.skill_pred_dim,
			net_arch=[],
			layer_norm=True,
			dropout=self.skill_dropout_prob,
			squash_output=False
		)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		x: jnp.ndarray,  # [b, l, d]
		context: jnp.ndarray,  # [b, l, d]
		mask: jnp.ndarray,  # [b, l]
		deterministic: bool = False,
		*args, **kwargs  # Do not remove this
	) -> Dict[str, jnp.ndarray]:
		context = self.encoder(context, mask, deterministic=deterministic)
		decoded_x = self.decoder(x, context, mask, deterministic=deterministic)

		pred_skills = self.pred_skills(decoded_x)  # [b, l, d]

		return {"pred_skills": pred_skills}
