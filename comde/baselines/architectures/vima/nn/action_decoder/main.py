from typing import Dict, List, Literal, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from comde.baselines.architectures.vima.nn.utils import build_mlp
from comde.utils.jax_utils.type_aliases import Activation


class ActionDecoder(nn.Module):
    action_dims: Dict[str, Union[int, List[int]]]
    hidden_dim: int
    hidden_depth: int
    activation: Union[str, Activation] = "relu"
    norm_type: Optional[Literal["batchnorm", "layernorm"]] = None
    last_layer_gain: Optional[float] = 0.01

    def setup(self) -> None:
        for key, action_dim in self.action_dims.items():
            assert (
                isinstance(action_dim, int) or isinstance(action_dim, list)
            ), f"action_dim for key {key} must be int or list[int], got {action_dim}"

        self.decoders = {
            key: (
                CategoricalNet(
                    action_dim=action_dim,
                    hidden_dim=self.hidden_dim,
                    hidden_depth=self.hidden_depth,
                    activation=self.activation,
                    norm_type=self.norm_type,
                    last_layer_gain=self.last_layer_gain,
                )
                if isinstance(action_dim, int)
                else MultiCategoricalNet(
                    action_dims=action_dim,
                    hidden_dim=self.hidden_dim,
                    hidden_depth=self.hidden_depth,
                    activation=self.activation,
                    norm_type=self.norm_type,
                    last_layer_gain=self.last_layer_gain,
                )
            )
            for key, action_dim in self.action_dims.items()
        }

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {key: decoder(x) for key, decoder in self.decoders.items()}


def _build_mlp_distribution_net(
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: Union[str, Activation] = "relu",
    norm_type: Optional[Literal["batchnorm", "layernorm"]] = None,
    last_layer_gain: Optional[float] = 0.01,
):
    """
    Use orthogonal initialization to initialize the MLP policy

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to have the
            Gaussian centered around 0.0 in the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    mlp = build_mlp(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        activation=activation,
        kernel_init="orthogonal",
        bias_init="zeros",
        norm_type=norm_type,
    )
    if last_layer_gain:
        assert last_layer_gain > 0
        mlp.layers[-1].kernel_init = nn.initializers.orthogonal(scale=last_layer_gain)
    return mlp


class CategoricalNet(nn.Module):
    """
    Use orthogonal initialization to initialize the MLP policy

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to make the
            Categorical close to uniform random at the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    action_dim: int
    hidden_dim: int
    hidden_depth: int
    activation: Union[str, Activation] = "relu"
    norm_type: Optional[Literal["batchnorm", "layernorm"]] = None
    last_layer_gain: Optional[float] = 0.01

    def setup(self) -> None:
        self.mlp = _build_mlp_distribution_net(
            output_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=self.hidden_depth,
            activation=self.activation,
            norm_type=self.norm_type,
            last_layer_gain=self.last_layer_gain,
        )
        self.head = CategoricalHead()

    def __call__(self, x):
        return self.head(self.mlp(x))


class MultiCategoricalNet(nn.Module):
    """
    Use orthogonal initialization to initialize the MLP policy
    Split head, does not share the NN weights

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to make the
            Categorical close to uniform random at the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    action_dims: List[int]
    hidden_dim: int
    hidden_depth: int
    activation: Union[str, Activation] = "relu"
    norm_type: Optional[Literal["batchnorm", "layernorm"]] = None
    last_layer_gain: Optional[float] = 0.01

    def setup(self) -> None:
        self.mlps = [
            _build_mlp_distribution_net(
                output_dim=action_dim,
                hidden_dim=self.hidden_dim,
                hidden_depth=self.hidden_depth,
                activation=self.activation,
                norm_type=self.norm_type,
                last_layer_gain=self.last_layer_gain,
            )
            for action_dim in self.action_dims
        ]
        self.head = MultiCategoricalHead(self.action_dims)

    def __call__(self, x):
        return self.head(jnp.concatenate([mlp(x) for mlp in self.mlps], axis=-1))


class CategoricalHead(nn.Module):
    def setup(self) -> None:
        self.categorical_rng = self.make_rng("dist")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.random.categorical(self.categorical_rng, logits=x)


class MultiCategoricalHead(nn.Module):
    action_dims: List[int]

    def setup(self) -> None:
        self._action_dims = tuple(self.action_dims)
        self.categorical_rngs = [
            self.make_rng("dist") for _ in range(len(self._action_dims))
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([
            jax.random.categorical(self.categorical_rngs[i], logits=split)
            for i, split in enumerate(jnp.split(x, self._action_dims, axis=-1))
        ])
