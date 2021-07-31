from box_embeddings.modules.regularization import TFL2SideBoxRegularizer
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
import tensorflow as tf

eps = 1e-23


def test_l2_side_regularizer():
    box = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float64)
    )
    regularizer = TFL2SideBoxRegularizer(weight=0.1)

    z = box.z  # (..., box_dim)
    Z = box.Z  # (..., box_dim)

    expected = 0.1 * tf.math.reduce_sum((Z - z) ** 2)
    res = regularizer(box)
    assert res == expected


def test_l2_side_regularizer_mean_reduction():
    box = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float64)
    )
    regularizer = TFL2SideBoxRegularizer(weight=0.1, reduction='mean')

    z = box.z  # (..., box_dim)
    Z = box.Z  # (..., box_dim)

    expected = 0.1 * tf.reduce_mean((Z - z) ** 2)
    res = regularizer(box)
    assert res == expected


def test_l2_side_regularizer_log():
    box = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float64)
    )
    regularizer = TFL2SideBoxRegularizer(weight=0.1, log_scale=True)

    z = box.z  # (..., box_dim)
    Z = box.Z  # (..., box_dim)

    expected = 0.1 * tf.math.reduce_sum(tf.math.log(tf.math.abs(Z - z) + eps))
    res = regularizer(box)
    assert res == expected
