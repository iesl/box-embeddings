import torch
from torch.nn.functional import softplus
from box_embeddings.modules.volume.hard_volume import HardVolume
from box_embeddings.modules.volume.soft_volume import SoftVolume, eps
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.common.testing.test_case import BaseTestCase
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, booleans
import numpy as np


@hypothesis.given(
    inp1=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    inp2=arrays(
        shape=(2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    volume_temperature=floats(1.0, 50.0),
    log_scale=booleans(),
)
@hypothesis.settings(max_examples=100, verbosity=hypothesis.Verbosity.verbose)
def test_volume(
    inp1: np.ndarray,
    inp2: np.ndarray,
    volume_temperature: float,
    log_scale: bool,
) -> None:
    inp1[..., 1] = np.absolute(inp1[..., 1]) + inp1[..., 0]  # make sure Z >z
    inp2[..., 1] = np.absolute(inp2[..., 1]) + inp2[..., 0]  # make sure Z >z
    box1 = BoxTensor(torch.tensor(inp1))
    box2 = BoxTensor(torch.tensor(inp2))
    soft_volume1 = SoftVolume(
        log_scale=log_scale, volume_temperature=volume_temperature
    )(box1)
    expected_volume = softplus(box1.Z - box1.z, beta=1 / volume_temperature)
    if not log_scale:
        expected_volume = torch.prod(expected_volume, dim=-1)
    else:
        expected_volume = torch.sum(torch.log(expected_volume + eps), dim=-1)

    if not log_scale:
        assert (soft_volume1 >= 0).all(), "soft volume greater than zero"

    assert torch.allclose(soft_volume1, expected_volume, rtol=1e-4)
