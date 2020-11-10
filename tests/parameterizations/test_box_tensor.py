from box_embeddings.parameterizations.box_tensor import BoxTensor
import torch
import numpy as np
import pytest
import hypothesis
from hypothesis.strategies import sampled_from


def test_simple_creation() -> None:
    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = BoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = BoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)


def test_shape_validation_during_creation():
    tensor = torch.tensor(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = BoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = BoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = BoxTensor(tensor)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box = BoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)
    assert box.data is None


def test_creation_from_vector():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    delta = torch.tensor(np.random.rand(*shape))
    v = torch.cat((z, z + delta.abs()), dim=-1)
    box = BoxTensor.from_vector(v)
    assert box.Z.shape == (3, 1, 5)


@hypothesis.given(
    sample=sampled_from(
        [
            ((4, 5, 10), (2, 10), (10,), (1, 1, 10)),
            ((4, 5, 10), (2, 3), (3,), ValueError),
            ((4, 5, 10), (4, 2, 2, 3), (4, 2, 3,), ValueError),
            ((4, 5, 10), (4, 2, 10), (4, 10,), (4, 1, 10)),
            ((4, 5, 10), (5, 2, 10), (5, 10,), (1, 5, 10)),
            ((4, 5, 10), (4, 2, 2, 2, 3), (4, 2, 2, 3,), ValueError),
            ((1, 5, 10), (5, 1, 2, 10), (5, 1, 10), (5, 1, 10)),
            ((5, 1, 10), (1, 5, 2, 10), (1, 5, 10), (1, 5, 10)),
            ((5, 1, 10), (5, 5, 2, 10), (5, 5, 10), (5, 5, 10)),
            ((5, 5, 10), (5, 5, 2, 10), (5, 5, 10), (5, 5, 10)),
        ]
    )
)
def test_broadcasting(sample):
    target_shape, input_data_shape, self_shape, expected = sample
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape

    if isinstance(expected, tuple):
        box.broadcast(target_shape)
        assert box.box_shape == expected
    else:
        with pytest.raises(expected):
            box.broadcast(target_shape)


# def test_reshape1():
#    target_shape = (-1, 10)
#    input_data_shape, self_shape = (5, 2, 10), (5, 10)
#    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
#    assert box.box_shape == self_shape
#    box.box_reshape(target_shape)
#    assert box.box_shape == (5, 10)
#


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
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape

    if expected == RuntimeError:
        with pytest.raises(expected):
            box.box_reshape(target_shape)
    else:
        new = box.box_reshape(target_shape)
        assert new.box_shape == expected
