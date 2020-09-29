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
