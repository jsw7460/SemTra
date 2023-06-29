from typing import cast

import flax.linen as nn
import jax.numpy as jnp
from transformers.models.t5.modeling_flax_t5 import FlaxT5Model


class WordEmbedding(nn.Module):
    def setup(self) -> None:
        model = cast(FlaxT5Model, FlaxT5Model.from_pretrained("t5-base"))
        self.embed_weight = cast(jnp.ndarray, model.params["shared"]["embedding"])
        self.embed_dim = self.embed_weight.shape[-1]
        self.embedding = nn.Embed(
            num_embeddings=self.embed_weight.shape[0],
            features=self.embed_weight.shape[-1],
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        
        self.output_dim = self.embed_dim
        del model

    def init(self, *args, **kwargs):
        result = super().init(*args, **kwargs)
        return result

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.embedding(x)
