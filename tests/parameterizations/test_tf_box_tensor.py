from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
import tensorflow as tf
import numpy as np
import pytest
import hypothesis
from hypothesis.strategies import sampled_from


def test_simple_creation() -> None:
    tensor = tf.constant(np.random.rand(3, 2, 3))
    box_tensor = TFBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)
    tensor = tf.constant(np.random.rand(2, 10))
    box_tensor = TFBoxTensor(tensor)
    assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, TFBoxTensor)


def test_shape_validation_during_creation():
    tensor = tf.constant(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = TFBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = TFBoxTensor(tensor)
    tensor = tf.constant(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = TFBoxTensor(tensor)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    Z = z + tf.constant(np.random.rand(*shape))
    print(z, Z)
    box = TFBoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)
    assert box.data is None


def test_creation_from_vector():
    shape = (3, 1, 5)
    z = tf.constant(np.random.rand(*shape))
    delta = tf.constant(np.random.rand(*shape))
    v = tf.concat([z, z + delta], axis=-1)
    box = TFBoxTensor.from_vector(v)
    assert box.Z.shape == (3, 1, 5)


@hypothesis.given(
    sample=sampled_from(
        [
            ((4, 5, 10), (2, 10), (10,), (1, 1, 10)),
            ((4, 5, 10), (2, 3), (3,), ValueError),
            (
                (4, 5, 10),
                (4, 2, 2, 3),
                (
                    4,
                    2,
                    3,
                ),
                ValueError,
            ),
            (
                (4, 5, 10),
                (4, 2, 10),
                (
                    4,
                    10,
                ),
                (4, 1, 10),
            ),
            (
                (4, 5, 10),
                (5, 2, 10),
                (
                    5,
                    10,
                ),
                (1, 5, 10),
            ),
            (
                (4, 5, 10),
                (4, 2, 2, 2, 3),
                (
                    4,
                    2,
                    2,
                    3,
                ),
                ValueError,
            ),
            ((1, 5, 10), (5, 1, 2, 10), (5, 1, 10), (5, 1, 10)),
            ((5, 1, 10), (1, 5, 2, 10), (1, 5, 10), (1, 5, 10)),
            ((5, 1, 10), (5, 5, 2, 10), (5, 5, 10), (5, 5, 10)),
            ((5, 5, 10), (5, 5, 2, 10), (5, 5, 10), (5, 5, 10)),
        ]
    )
)
def test_broadcasting(sample):
    target_shape, input_data_shape, self_shape, expected = sample
    box = TFBoxTensor(tf.constant(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    print(box)
    if isinstance(expected, tuple):
        box.broadcast(target_shape)
        assert box.box_shape == expected
    else:
        with pytest.raises(expected):
            box.broadcast(target_shape)


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
