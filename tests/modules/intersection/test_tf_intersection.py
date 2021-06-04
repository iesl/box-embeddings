from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.modules.intersection.tf_intersection import TFIntersection
from box_embeddings.modules.intersection.tf_hard_intersection import (
    TFHardIntersection,
)
from box_embeddings.modules.intersection.tf_gumbel_intersection import (
    TFGumbelIntersection,
)
import tensorflow as tf


def test_hard_intersection() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]], dtype=tf.float32)
    )
    box2 = TFBoxTensor(
        tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]], dtype=tf.float32)
    )
    expected = TFHardIntersection()(box1, box2)
    res = TFIntersection()(box1, box2)
    assert res == expected


def test_gumbel_intersection() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]], dtype=tf.float32)
    )
    box2 = TFBoxTensor(
        tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]], dtype=tf.float32)
    )
    expected = TFGumbelIntersection()(box1, box2)
    res = TFIntersection(intersection_temperature=1.0)(box1, box2)
    assert res == expected
