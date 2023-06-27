from __future__ import annotations

import warnings
from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

Float3D = Tuple[float, float, float]


@jax.jit
def basic_image_array_preprocess(
    img: jnp.ndarray,
    mean: Float3D = (0.5, 0.5, 0.5),
    std: Float3D = (0.5, 0.5, 0.5),
    shape: Tuple[int, int] | None = None,
):
    """
    Check for resize, and divide by 255
    """

    assert isinstance(img, jnp.ndarray)
    assert img.ndim >= 4
    original_shape = list(img.shape)
    img = img.astype(jnp.float32)
    img = img.reshape(-1, *img.shape[-3:])
    assert img.ndim == 4

    input_size = img.shape[-2:]
    assert img.max() > 2, "img should be between [0, 255] before normalize"

    if shape and input_size != shape:
        warnings.warn(
            f'{"Down" if shape < input_size else "Up"}sampling image'
            f" from original resolution {input_size}x{input_size}"
            f" to {shape}x{shape}"
        )
        img = jax.image.resize(img, shape, "bilinear")
        img = lax.clamp(img, 0.0, 255.0)

    B, C, H, W = img.shape
    assert C % 3 == 0, "channel must divide 3"
    img = img.reshape(B * C // 3, 3, H, W)
    img = jax_normalize(img / 255.0, mean=mean, std=std)
    original_shape[-2:] = H, W
    return img.view(original_shape)


def jax_normalize(
    x: jnp.ndarray, mean: Float3D, std: Float3D, inplace: bool = False
) -> jnp.ndarray:
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        x (jnp.ndarray): Image array of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        jnp.ndarray: Normalized image array.
    """
    if not isinstance(x, jnp.ndarray):
        raise TypeError("array should be a jax numpy array. Got {}.".format(type(x)))

    if not inplace:
        x = x.copy()

    dtype = x.dtype
    _mean: jnp.ndarray = jnp.asarray(mean, dtype=dtype)
    _std: jnp.ndarray = jnp.array(std, dtype=dtype)

    if (_std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    if _mean.ndim == 1:
        _mean = _mean[:, None, None]
    if _std.ndim == 1:
        _std = _std[:, None, None]
    return (x - _mean) / _std
