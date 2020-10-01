from box_embeddings.parameterizations.box_tensor import BoxTensor
import torch
import numpy as np
import pytest


def test_simple_creation() -> None:
    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = BoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = BoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
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
    assert box.data.shape == (3, 1, 2, 5)


def test_creation_from_vector():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    delta = torch.tensor(np.random.rand(*shape))
    v = torch.cat((z, z + delta.abs()), dim=-1)
    box = BoxTensor.from_vector(v)
    assert box.data.shape == (3, 1, 2, 5)


def test_broadcasting1():
    target_shape = (4, 5, 10)
    # 1
    input_data_shape, self_shape = (2, 10), (10,)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    box.broadcast(target_shape)
    assert box.box_shape == (1, 1, 10)


def test_broadcasting2():
    target_shape = (4, 5, 10)
    # 2
    input_data_shape, self_shape = (2, 3), (3,)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    with pytest.raises(ValueError):
        box.broadcast(target_shape)


def test_broadcasting3():
    target_shape = (4, 5, 10)
    # 3
    input_data_shape, self_shape = (4, 2, 2, 3), (4, 2, 3,)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    with pytest.raises(ValueError):
        box.broadcast(target_shape)


def test_broadcasting4():
    target_shape = (4, 5, 10)
    # 4
    input_data_shape, self_shape = (4, 2, 10), (4, 10)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    box.broadcast(target_shape)
    assert box.box_shape == (4, 1, 10)


def test_broadcasting5():
    # 5
    target_shape = (4, 5, 10)
    input_data_shape, self_shape = (5, 2, 10), (5, 10)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    box.broadcast(target_shape)
    assert box.box_shape == (1, 5, 10)


def test_broadcasting6():
    # 5
    target_shape = (4, 5, 7, 8, 10)
    input_data_shape, self_shape = (4, 7, 2, 10), (4, 7, 10)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    box.broadcast(target_shape)
    assert box.box_shape == (4, 1, 7, 1, 10)


def test_broadcasting7():
    target_shape = (4, 5, 10)
    # 3
    input_data_shape, self_shape = (4, 2, 2, 2, 3), (4, 2, 2, 3,)
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape
    with pytest.raises(ValueError):
        box.broadcast(target_shape)
