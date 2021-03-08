from box_embeddings.parameterizations import TFBoxTensor, TFSigmoidBoxTensor
from box_embeddings.common.tf_utils import inv_sigmoid
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
    box_tensor = TFSigmoidBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)
    tensor = tf.constant(np.random.rand(2, 10))
    box_tensor = TFSigmoidBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)


def test_shape_validation_during_creation():
    tensor = tf.constant(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = TFSigmoidBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = TFSigmoidBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = TFSigmoidBoxTensor(tensor)


def test_W_from_zZ():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    Z = z + tf.constant(np.random.rand(*shape))
    box_W = TFSigmoidBoxTensor.W(z, Z)
    eps = 1e-07
    tf.clip_by_valu

    w1 = inv_sigmoid(
        tf.clip_by_value(z, clip_value_min=eps, clip_value_max=eps)
    )
    w2 = inv_sigmoid(
        tf.clip_by_value(
            ((Z - z) / (1.0 - z)), clip_value_min=eps, clip_value_max=eps
        )
    )

    W = tf.stack((w1, w2), -2)
    assert tf.debugging.assert_near(box_W, W, rtol=1e-05, atol=1e-08)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    Z = z + tf.constant(np.random.rand(*shape))
    box = TFSigmoidBoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)


def test_creation_from_vector():
    shape = (3, 1, 5)
    w1 = tf.constant(np.random.rand(*shape))
    w2 = tf.constant(np.random.rand(*shape))
    v = tf.concat((w1, w2), dim=-1)
    box = TFSigmoidBoxTensor.from_vector(v)
    assert box.Z.shape == (3, 1, 5)
    assert tf.debugging.assert_near(
        box.z, tf.math.sigmoid(w1), rtol=1e-05, atol=1e-08
    )
    assert tf.debugging.assert_near(
        box.Z,
        tf.math.sigmoid(w1)
        + tf.math.sigmoid(w2) * (1.0 - tf.math.sigmoid(w1)),
        rtol=1e-05,
        atol=1e-08,
    )
