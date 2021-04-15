from box_embeddings.parameterizations import BoxTensor, SigmoidBoxTensor
from box_embeddings.common.utils import inv_sigmoid
import torch
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
    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = SigmoidBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = SigmoidBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)


def test_shape_validation_during_creation():
    tensor = torch.tensor(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = SigmoidBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = SigmoidBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = SigmoidBoxTensor(tensor)


def test_W_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box_W = SigmoidBoxTensor.W(z, Z)
    eps = torch.finfo(z.dtype).tiny
    w1 = inv_sigmoid(z.clamp(eps, 1.0 - eps))
    w2 = inv_sigmoid(((Z - z) / (1.0 - z)).clamp(eps, 1.0 - eps))
    W = torch.stack((w1, w2), -2)
    assert torch.allclose(box_W, W)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box = SigmoidBoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)


def test_creation_from_vector():
    shape = (3, 1, 5)
    w1 = torch.tensor(np.random.rand(*shape))
    w2 = torch.tensor(np.random.rand(*shape))
    v = torch.cat((w1, w2), dim=-1)
    box = SigmoidBoxTensor.from_vector(v)
    assert box.Z.shape == (3, 1, 5)
    assert torch.allclose(box.z, torch.sigmoid(w1))
    assert torch.allclose(
        box.Z,
        torch.sigmoid(w1) + torch.sigmoid(w2) * (1.0 - torch.sigmoid(w1)),
    )
