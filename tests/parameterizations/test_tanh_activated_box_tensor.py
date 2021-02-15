from box_embeddings.parameterizations import BoxTensor, TanhActivatedBoxTensor
import box_embeddings.common.constant as constant

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
    box_tensor = TanhActivatedBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = TanhActivatedBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)


def test_shape_validation_during_creation():
    tensor = torch.tensor(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = TanhActivatedBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = TanhActivatedBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = TanhActivatedBoxTensor(tensor)


def test_W_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box_W = TanhActivatedBoxTensor.W(z, Z)
    tanh_eps = constant.TANH_EPS
    z_ = z.clamp(0.0, 1.0 - tanh_eps / 2.0)
    Z_ = Z.clamp(tanh_eps / 2.0, 1.0)
    w1 = 2 * z_ - 1
    w2 = 2 * (Z_ - z_) / (1.0 - z_) - 1
    W = torch.stack((w1, w2), -2)
    assert torch.allclose(box_W, W)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box = TanhActivatedBoxTensor.from_zZ(z, Z)
    assert box.z.shape == (3, 1, 5)


def test_creation_from_vector():
    shape = (3, 1, 5)
    tanh_eps = constant.TANH_EPS
    w1 = torch.tensor(np.random.rand(*shape))
    w2 = torch.tensor(np.random.rand(*shape))
    v = torch.cat((w1, w2), dim=-1)
    box = TanhActivatedBoxTensor.from_vector(v)
    z = (w1.clamp(-1, 1.0 - tanh_eps) + 1) / 2
    Z = z + (w2.clamp(-1.0 + tanh_eps, 1) + 1) * (1.0 - z) / 2
    assert box.Z.shape == (3, 1, 5)
    assert torch.allclose(box.z, z)
    assert torch.allclose(box.Z, Z)


# def test_warning_in_creation_from_zZ():
#    shape = (3, 1, 5)
#    z = torch.tensor(np.random.rand(*shape))
#    Z = z + torch.tensor(np.random.rand(*shape))
#    with pytest.warns(UserWarning):
#        box = TanhActivatedBoxTensor.from_zZ(z, Z)
