from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

from comde.baselines.architectures.vima.nn.utils import build_mlp, identity


class ActionEmbedding(nn.Module):
    output_dim: int
    embed_dict: Dict[str, nn.Module]

    def setup(self) -> None:
        embed_dict_output_dim = sum(
            module.output_dim for module in self.embed_dict.values()
        )
        self.post_layer = (
            identity if self.output_dim == embed_dict_output_dim
            else nn.Dense(self.output_dim)
        )
        self._input_fields_checked = False

    def forward(self, x_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        if not self._input_fields_checked:
            assert set(x_dict.keys()) == set(self.embed_dict.keys())
            self._input_fields_checked = True
        return self.post_layer(
            jnp.concatenate(
                [
                    module(x_dict[name])
                    for name, module in sorted(self.embed_dict.items(), key=lambda x: x[0])
                ],
                axis=-1,
            )
        )


class ContinuousActionEmbedding(nn.Module):
    output_dim: int
    hidden_dim: int
    hidden_depth: int

    def setup(self) -> None:
        self._layer = build_mlp(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            hidden_depth=self.hidden_depth,
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._layer(x)
