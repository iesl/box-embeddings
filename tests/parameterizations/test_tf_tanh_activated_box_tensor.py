from box_embeddings.parameterizations import TFBoxTensor, TFTanhBoxTensor
import box_embeddings.common.constant as constant

import tensorflow as tf
import numpy as np
import pytest
import warnings
from hypothesis.extra.numpy import arrays
import hypothesis
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    fixed_dictionaries,
    just,
)


def test_simple_creation() -> None:
    tensor = tf.constant(np.random.rand(3, 2, 3))
    box_tensor = TFTanhBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)
    tensor = tf.constant(np.random.rand(2, 10))
    box_tensor = TFTanhBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)


def test_shape_validation_during_creation():
    tensor = tf.constant(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = TFTanhBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = TFTanhBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = TFTanhBoxTensor(tensor)


def test_W_from_zZ():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    Z = z + tf.constant(np.random.rand(*shape))
    box_W = TFTanhBoxTensor.W(z, Z)
    tanh_eps = constant.TANH_EPS

    z_ = tf.clip_by_value(
        z, clip_value_min=0.0, clip_value_max=1.0 - tanh_eps / 2.0
    )
    Z_ = tf.clip_by_value(Z, clip_value_min=tanh_eps / 2.0, clip_value_max=1.0)

    w1 = 2 * z_ - 1
    w2 = 2 * (Z_ - z_) / (1.0 - z_) - 1
    W = tf.stack((w1, w2), -2)
    assert np.allclose(box_W, W)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    Z = z + tf.constant(np.random.rand(*shape))
    box = TFTanhBoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)


def test_creation_from_vector():
    shape = (3, 1, 5)
    tanh_eps = constant.TANH_EPS
    w1 = tf.constant(np.random.rand(*shape))
    w2 = tf.constant(np.random.rand(*shape))
    v = tf.concat((w1, w2), axis=-1)
    box = TFTanhBoxTensor.from_vector(v)

    z = (
        tf.clip_by_value(w1, clip_value_min=-1, clip_value_max=1.0 - tanh_eps)
        + 1
    ) / 2
    Z = (
        z
        + (
            tf.clip_by_value(
                w2, clip_value_min=-1.0 + tanh_eps, clip_value_max=1
            )
            + 1
        )
        * (1.0 - z)
        / 2
    )

    assert box.Z.shape == (3, 1, 5)
    assert np.allclose(box.z, z)
    assert np.allclose(box.Z, Z)


# def test_warning_in_creation_from_zZ():
#    shape = (3, 1, 5)
#    z = torch.tensor(np.random.rand(*shape))
#    Z = z + torch.tensor(np.random.rand(*shape))
#    with pytest.warns(UserWarning):
#        box = TanhBoxTensor.from_zZ(z, Z)
