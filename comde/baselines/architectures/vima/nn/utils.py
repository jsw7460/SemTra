from __future__ import annotations

import math
from functools import partial
from typing import List, Literal, Optional, Union

import flax.linen as nn
import jax.numpy as jnp

from comde.utils.jax_utils.type_aliases import Activation, ModuleLike


class Embedding(nn.Embed):
    @property
    def output_dim(self):
        return self.embedding_dim


def identity(x: jnp.ndarray) -> jnp.ndarray:
    return x


def build_mlp(
    *,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: Optional[int] = None,
    num_layers: Optional[int] = None,
    activation: Union[str, Activation] = "relu",
    kernel_init: Union[str, nn.initializers.Initializer] = "orthogonal",
    bias_init: Union[str, nn.initializers.Initializer] = "zeros",
    norm_type: Optional[Literal["batchnorm", "layernorm"]] = None,
    add_input_activation: Union[bool, str, Activation] = False,
    add_input_norm: bool = False,
    add_output_activation: Union[bool, str, Activation] = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    """
    In other popular RL implementations, tanh is typically used with orthogonal
    initialization, which may perform better than ReLU.

    Args:
        norm_type: None, "batchnorm", "layernorm", applied to intermediate layers
        add_input_activation: whether to add a nonlinearity to the input _before_
            the MLP computation. This is useful for processing a feature from a preceding
            image encoder, for example. Image encoder typically has a linear layer
            at the end, and we don't want the MLP to immediately stack another linear
            layer on the input features.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_input_norm: see `add_input_activation`, whether to add a normalization layer
            to the input _before_ the MLP computation.
            values: True to add the `norm_type` to the input
        add_output_activation: whether to add a nonlinearity to the output _after_ the
            MLP computation.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_output_norm: see `add_output_activation`, whether to add a normalization layer
            _after_ the MLP computation.
            values: True to add the `norm_type` to the input
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    kernel_init = get_initializer(kernel_init, act_layer)
    bias_init = get_initializer(bias_init, act_layer)

    lower_norm_type = norm_type.lower() if norm_type else None

    if not lower_norm_type:
        _norm_type = lambda: identity
    elif lower_norm_type == "batchnorm":
        _norm_type = nn.BatchNorm
    elif lower_norm_type == "layernorm":
        _norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")

    hidden_depth = (
        num_layers - 1 if num_layers is not None and hidden_depth is None
        else hidden_depth
    )
    mods: List[ModuleLike]
    if hidden_depth is None or hidden_depth == 0:
        mods = [
            nn.Dense(output_dim, kernel_init=kernel_init, bias_init=bias_init)
        ]
    else:
        mods = [
            nn.Dense(hidden_dim, kernel_init=kernel_init, bias_init=bias_init),
            _norm_type(),
            act_layer,
        ]
        for _ in range(hidden_depth - 1):
            mods += [
                nn.Dense(hidden_dim, kernel_init=kernel_init, bias_init=bias_init),
                _norm_type(),
                act_layer,
            ]
        mods.append(nn.Dense(output_dim, kernel_init=kernel_init, bias_init=bias_init))

    if add_input_norm:
        mods = [_norm_type()] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer] + mods
    if add_output_norm:
        mods.append(_norm_type())
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer)

    return nn.Sequential(mods)


def get_activation(activation: Union[str, Activation, None]) -> Activation:
    if not activation or not isinstance(activation, str):
        return identity
    activation = activation.lower()
    assert hasattr(nn, activation), f"Not supported activations: {activation}"
    return getattr(nn, activation)


def get_initializer(
    method: Union[str, nn.initializers.Initializer],
    activation: Activation,
) -> nn.initializers.Initializer:
    if not isinstance(method, str):
        return method
    assert hasattr(nn.initializers, method), f"Initializer nn.initializers.{method} does not exist"
    if method == "orthogonal":
        try:
            gain = calculate_gain(activation)
        except ValueError:
            gain = 1.0
        return nn.initializers.orthogonal(scale=gain)
    return getattr(nn.initializers, method)


def calculate_gain(nonlinearity: Activation, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ["linear", "conv", "conv_transpose"]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
