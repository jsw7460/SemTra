from typing import Union, Tuple, List

from flax import linen as nn
from jax import numpy as jnp
from comde.comde_modules.common.utils import create_mlp


EPSILON = 1E-32

class PrimIntentEmbeddingVQ(nn.Module):
	"""

	Input:
		Skill, language guidance

	Output:
		Intended skill

	Procedure:
		1. Concatenate skill and langauge guidance
		2. Embed into clip dimension (codebook_dim in the following variable)
		3. Add it to semantic skill (Called intended skill)
	"""

	net_arch: List
	n_codebook: int
	codebook_dim: int  # Note: == skill (clip) dimension

	init_codebook: Union[jnp.ndarray] = None

	skill_lang_embedding = None
	codebook = None

	def setup(self) -> None:
		self.skill_lang_embedding = create_mlp(
			output_dim=self.codebook_dim,
			net_arch=self.net_arch,
			squash_output=False
		)
		self.codebook = nn.Embed(
			self.n_codebook,
			self.codebook_dim,
			dtype=jnp.float64
		)

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def embeddnig_init(self, codebook: jnp.ndarray):
		"""Call this outside"""
		self.codebook.embedding_init(lambda *args, **kwargs: codebook)

	def forward(
		self,
		skills: jnp.ndarray,	# [b, l, d] or [b, d]
		language_operators: jnp.ndarray,	# [b, l, d] or [b, d]
		deterministic: bool = False,
		training: bool = True
	):
		"""
			return: "Quantized" vectors
		"""
		unquantized_vec = self.get_unquantized_vec(
			skills=skills,
			language_operators=language_operators,
			deterministic=deterministic,
			training=training
		)	# [b, l, d] or [b, d]
		quantized_vec = self.get_quantized_vec(unquantized_vec=unquantized_vec)	# [b, l, d] or [b, d]
		# NOTE
		#	 				  Gradient
		#	Unquantized_vec		 O
		#	Quantized_vec		 X
		return unquantized_vec, quantized_vec

	def get_current_embedding(self):
		return self.codebook.embedding

	def get_unquantized_vec(
		self,
		skills: jnp.ndarray,	 # [b, l, d] or [b, d]
		language_operators: jnp.ndarray,	# [b, l, d] or [b, d]
		deterministic: bool = False,
		training: bool = True
	):
		"""
			return: "UNquantized" vectors
		"""
		skill_lang = jnp.concatenate((skills, language_operators), axis=-1)	# [b, l, 2d] or [b, 2d]
		sl_embedding = self.skill_lang_embedding(skill_lang, deterministic=deterministic, training=training)
		return sl_embedding

	def get_quantized_vec(
		self,
		unquantized_vec: jnp.ndarray	# [b, l, d]
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""
			From unquantized vector representation, return the quantized one.
			Want:
				sl: b d -> b 1 d
				sl:	b l d -> b l 1 d
				Codebook: 1 n d  //  1 1 n d

				subtract -> [b n d]  //  [b l n d]

		"""
		nearest_idxs, _ = self.get_nearest_code_idxs(vec=unquantized_vec)
		quantized_h = self.codebook(nearest_idxs)  # [b, l, d] or [b, d]
		return quantized_h

	def get_nearest_code_idxs(self, vec: jnp.ndarray):
		ndim = vec.ndim
		unquantized_vec = jnp.expand_dims(vec, axis=ndim - 1)  # [batch_size, l, 1, dim]
		codebook_embedding = jnp.expand_dims(
			self.codebook.embedding,
			axis=list(range(unquantized_vec.ndim - 2))
		)  # [1, 1, n_codebook, dim]
		distance = unquantized_vec - codebook_embedding  # [batch_size, l, n_codebook, dim]
		distance = jnp.linalg.norm(distance, axis=-1, keepdims=False)  # [b, l, n_codebook] or [b, n_codebook]
		# Note: The gradients vanishing due to argmin operation
		nearest_idxs = jnp.argmin(distance, axis=-1, keepdims=False)  # [b, l] or [b]
		return nearest_idxs, distance

	def get_mean_of_minibatch_wrt_idxs(self, unquantized_h: jnp.ndarray, codebook_idxs: jnp.ndarray) -> jnp.ndarray:
		"""
			:param unquantized_h: [batch_size, dim]
			:param codebook_idxs: [batch_size, ]

			return: [n_codebook, dim]: A target of moving average. So the codebook should be shifted to this direction.
		"""
		n_codebook = self.n_codebook
		mean_of_minibatch = jnp.empty((0, self.codebook_dim))
		for k in range(n_codebook):
			kth_idx = jnp.where(codebook_idxs == k, 1, 0)
			kth_mean = jnp.sum(unquantized_h * kth_idx.reshape(-1, 1), axis=0) / (jnp.sum(kth_idx) + EPSILON)
			mean_of_minibatch = jnp.vstack((mean_of_minibatch, kth_mean))
		return mean_of_minibatch
