import flax.linen as nn
import jax.numpy as jnp
from transformers.models.t5.modeling_flax_t5 import FlaxT5Model


class WordEmbedding(nn.Module):
    def setup(self) -> None:
        model: FlaxT5Model = FlaxT5Model.from_pretrained("t5-base")
        embed_weight: jnp.ndarray = model.params["shared"]["weight"]
        self.embed_dim = embed_weight.shape[-1]
        self.embedding = nn.Embed(
            num_embeddings=embed_weight.shape[0],
            features=embed_weight.shape[-1],
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.embedding.weight = embed_weight
        self.output_dim = self.embed_dim
        del model

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.embedding(x)
