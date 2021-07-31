import pytest

from box_embeddings.modules.regularization import TFBoxRegularizer
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
import tensorflow as tf


def test_regularizer():
    box = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float64)
    )
    regularizer = TFBoxRegularizer(weight=0.1)

    expected = NotImplementedError
    with pytest.raises(expected):
        regularizer(box)
