import tensorflow as tf
from box_embeddings.modules.volume.tf_volume import TFVolume
from box_embeddings.modules.volume.tf_soft_volume import TFSoftVolume, eps
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
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
    volume_temperature=floats(1.0, 50.0),
    log_scale=booleans(),
)
@hypothesis.settings(max_examples=100, verbosity=hypothesis.Verbosity.verbose)
def test_volume(
    inp1: np.ndarray,
    volume_temperature: float,
    log_scale: bool,
) -> None:
    inp1[..., 1] = np.absolute(inp1[..., 1]) + inp1[..., 0]  # make sure Z >z
    box1 = TFBoxTensor(tf.Variable(inp1))
    soft_volume1 = TFSoftVolume(
        log_scale=log_scale, volume_temperature=volume_temperature
    )(box1)
    expected_volume = tf.math.softplus(
        (box1.Z - box1.z) * (1 / volume_temperature)
    )
    if not log_scale:
        expected_volume = tf.math.reduce_prod(expected_volume, axis=-1)
    else:
        expected_volume = tf.math.reduce_sum(
            tf.math.log(expected_volume + eps), axis=-1
        )

    if not log_scale:
        assert tf.reduce_all(
            soft_volume1 >= 0
        ), "soft volume greater than zero"

    assert np.allclose(soft_volume1, expected_volume, rtol=1e-4)
