import tensorflow as tf
from box_embeddings.modules.volume.tf_bessel_volume import TFBesselApproxVolume
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
import numpy as np


def test_volume() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]], dtype=tf.float64)
    )
    box2 = TFBoxTensor(
        tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]], dtype=tf.float64)
    )
    volume_layer = TFBesselApproxVolume(log_scale=False)
    expected1 = tf.Variable([3.490526, 1.4467278], dtype=tf.float64)
    expected2 = tf.Variable([3.490526, 0.7444129], dtype=tf.float64)
    res1 = volume_layer(box1)
    res2 = volume_layer(box2)
    assert np.allclose(res1, expected1, rtol=1e-4)
    assert np.allclose(res2, expected2, rtol=1e-4)


def test_log_volume() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]], dtype=tf.float64)
    )
    box2 = TFBoxTensor(
        tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]], dtype=tf.float64)
    )
    volume_layer = TFBesselApproxVolume(log_scale=True)
    expected1 = tf.Variable([1.25004, 0.36924], dtype=tf.float64)
    expected2 = tf.Variable([1.25004, -0.29517], dtype=tf.float64)
    res1 = volume_layer(box1)
    res2 = volume_layer(box2)
    assert np.allclose(res1, expected1, rtol=1e-4)
    assert np.allclose(res2, expected2, rtol=1e-4)
