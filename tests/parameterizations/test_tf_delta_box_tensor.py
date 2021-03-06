from box_embeddings.parameterizations import TFBoxTensor, TFMinDeltaBoxTensor
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
    box_tensor = TFMinDeltaBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)
    tensor = tf.constant(np.random.rand(2, 10))
    box_tensor = TFMinDeltaBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)


def test_shape_validation_during_creation():
    tensor = tf.constant(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = TFMinDeltaBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = TFMinDeltaBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = TFMinDeltaBoxTensor(tensor)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    Z = z + tf.constant(np.random.rand(*shape))
    box = TFMinDeltaBoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)


@hypothesis.given(
    beta=floats(1.0, 50.0),
    threshold=integers(20, 50),
)
def test_creation_from_vector(beta, threshold):
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    w_delta = tf.constant(np.random.rand(*shape))
    v = tf.concat([z, w_delta], -1)
    box = TFMinDeltaBoxTensor.from_vector(v, beta=beta, threshold=threshold)
    assert box.Z.shape == (3, 1, 5)


@hypothesis.given(
    sample=sampled_from(
        [
            ((-1, 10), (5, 2, 10), (5, 10), (5, 10)),
            ((-1, 10), (5, 4, 2, 10), (5, 4, 10), (20, 10)),
            ((10, 2, 10), (20, 2, 10), (20, 10), (10, 2, 10)),
            ((-1, 10), (2, 5), (5,), RuntimeError),
            ((2, 10), (5, 2, 10), (5, 10), RuntimeError),
        ]
    )
)
def test_reshape(sample):
    target_shape, input_data_shape, self_shape, expected = sample
    box = TFBoxTensor(tf.constant(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape

    if expected == RuntimeError:
        with pytest.raises(expected):
            box.box_reshape(target_shape)
    else:
        new = box.box_reshape(target_shape)
        assert new.box_shape == expected
