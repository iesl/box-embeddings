from box_embeddings.parameterizations import BoxTensor, MinDeltaBoxTensor
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
    box_tensor = MinDeltaBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = MinDeltaBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
    assert isinstance(box_tensor, BoxTensor)


def test_shape_validation_during_creation():
    tensor = torch.tensor(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = MinDeltaBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = MinDeltaBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = MinDeltaBoxTensor(tensor)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box = MinDeltaBoxTensor.from_zZ(z, Z)
    assert box.data.shape == (3, 1, 2, 5)


@hypothesis.given(
    beta=floats(1.0, 50.0), threshold=integers(20, 50),
)
def test_creation_from_vector(beta, threshold):
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    w_delta = torch.tensor(np.random.rand(*shape))
    v = torch.cat((z, w_delta), dim=-1)
    box = MinDeltaBoxTensor.from_vector(v, beta=beta, threshold=threshold)
    assert box.data.shape == (3, 1, 2, 5)
    assert torch.allclose(box.z, z)
    assert torch.allclose(
        box.Z,
        z
        + torch.nn.functional.softplus(
            w_delta, beta=beta, threshold=threshold
        ),
    )


# def test_warning_in_creation_from_zZ():
#    shape = (3, 1, 5)
#    z = torch.tensor(np.random.rand(*shape))
#    Z = z + torch.tensor(np.random.rand(*shape))
#    with pytest.warns(UserWarning):
#        box = MinDeltaBoxTensor.from_zZ(z, Z)
