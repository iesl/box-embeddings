from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.volume.hard_volume import HardVolume
from box_embeddings.modules.volume.soft_volume import SoftVolume
from box_embeddings.modules.volume.bessel_volume import BesselApproxVolume
import torch
import numpy as np

import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, booleans


@hypothesis.given(
    inp=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    log_scale=booleans(),
)
def test_hard_volume(inp, log_scale) -> None:
    box = BoxTensor(torch.tensor(inp))
    expected = HardVolume(log_scale)(box)
    res = Volume(log_scale)(box)
    assert torch.allclose(res, expected)


@hypothesis.given(
    inp=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    volume_temperature=floats(1.0, 50.0),
    log_scale=booleans(),
)
def test_soft_volume(inp, volume_temperature, log_scale) -> None:
    box = BoxTensor(torch.tensor(inp))
    expected = SoftVolume(log_scale, volume_temperature)(box)
    res = Volume(volume_temperature=volume_temperature, log_scale=log_scale)(
        box
    )
    assert torch.allclose(res, expected)


@hypothesis.given(
    inp=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    volume_temperature=floats(1.0, 50.0),
    intersection_temperature=floats(1.0, 50.0),
    log_scale=booleans(),
)
def test_bessel_volume(
    inp, volume_temperature, intersection_temperature, log_scale
) -> None:
    box = BoxTensor(torch.tensor(inp))
    expected = BesselApproxVolume(
        log_scale, volume_temperature, intersection_temperature
    )(box)
    res = Volume(
        intersection_temperature=intersection_temperature,
        volume_temperature=volume_temperature,
        log_scale=log_scale,
    )(box)
    assert torch.allclose(res, expected)
