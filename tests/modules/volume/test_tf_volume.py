from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.modules.volume.tf_volume import TFVolume
from box_embeddings.modules.volume.tf_hard_volume import TFHardVolume
from box_embeddings.modules.volume.tf_soft_volume import TFSoftVolume
from box_embeddings.modules.volume.tf_bessel_volume import TFBesselApproxVolume
import tensorflow as tf
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
    box = TFBoxTensor(tf.Variable(inp))
    expected = TFHardVolume(log_scale)(box)
    res = TFVolume(log_scale)(box)
    assert np.allclose(res, expected)


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
    box = TFBoxTensor(tf.Variable(inp))
    expected = TFSoftVolume(log_scale, volume_temperature)(box)
    res = TFVolume(volume_temperature=volume_temperature, log_scale=log_scale)(
        box
    )
    assert np.allclose(res, expected)


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
    box = TFBoxTensor(tf.Variable(inp))
    expected = TFBesselApproxVolume(
        log_scale, volume_temperature, intersection_temperature
    )(box)
    res = TFVolume(
        intersection_temperature=intersection_temperature,
        volume_temperature=volume_temperature,
        log_scale=log_scale,
    )(box)
    assert np.allclose(res, expected)
