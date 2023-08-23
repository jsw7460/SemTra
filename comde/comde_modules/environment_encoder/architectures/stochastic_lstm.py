from flax import linen as nn
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.environment_encoder.architectures.lstm import LSTM

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 1
LOG_STD_MIN = -10


class StochasticLSTM(nn.Module):
	"""
		Hidden state of lstm is sampled by a normal distribution
	"""

	lstm: LSTM

	mu = None
	log_std = None

	def setup(self) -> None:
		self.mu = create_mlp(self.lstm.hidden_dim, [])
		self.log_std = create_mlp(self.lstm.hidden_dim, [])

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		sequence: jnp.ndarray,
		n_iter: jnp.ndarray,
		deterministic: bool = False,
		training: bool = True,
	):
		rng = self.make_rng("sampling")
		batch_size = sequence.shape[0]
		stacked_output, _ = self.lstm(
			sequence=sequence,
			batch_size=batch_size,
			deterministic=deterministic
		)  # stacked_output: [batch_size, subseq_len + 1, dim]

		x = stacked_output.at[jnp.arange(batch_size), n_iter].get()  # [batch_size, dim]

		mu, log_std = self.get_distribution_parameters(x, deterministic=deterministic, training=training)
		hidden_variables_dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_std))
		hidden_variables = hidden_variables_dist.sample(seed=rng)

		return stacked_output, hidden_variables

	def get_distribution_parameters(self, x: jnp.ndarray, deterministic: bool = False, training: bool = True):
		mu = self.mu(x, deterministic=deterministic, training=training)
		log_std = self.log_std(x, deterministic=deterministic, training=training)
		log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
		return mu, log_std

	def get_output_and_distributions_from_input(
		self,
		sequence: jnp.ndarray,
		batch_size: int,
		n_iter: jnp.ndarray,
		deterministic: bool = False,
		training: bool = True
	):
		stacked_output, _ = self.lstm(
			sequence=sequence,
			batch_size=batch_size,
			deterministic=deterministic
		)  # stacked_output: [batch_size, subseq_len + 1, dim]

		x = stacked_output.at[jnp.arange(batch_size), n_iter].get()  # [batch_size, dim]

		mu, log_std = self.get_distribution_parameters(x, deterministic=deterministic, training=training)
		hidden_variables_dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_std))
		return stacked_output, hidden_variables_dist
