from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
from box_embeddings.modules.intersection.tf_hard_intersection import (
    tf_hard_intersection,
    TFHardIntersection,
)
import tensorflow as tf


def test_intersection() -> None:
    box1 = TFBoxTensor(tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]))
    box2 = TFBoxTensor(tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]))
    res = TFBoxTensor(tf.Variable([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]))
    assert res == tf_hard_intersection(box1, box2)


def test_intersection_with_broadcasting() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]])
    )  # box_shape (2,2)
    box2 = TFBoxTensor(tf.Variable([[2, 0], [6, 2]]))  # box_shape (2,)
    res = TFBoxTensor(tf.Variable([[[2, 1], [3, 2]], [[2, 1], [3, 2]]]))
    assert res == tf_hard_intersection(box1, box2)
    assert res == tf_hard_intersection(box2, box1)


def test_intersection_module() -> None:
    box1 = TFBoxTensor(tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]))
    box2 = TFBoxTensor(tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]))
    res = TFBoxTensor(tf.Variable([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]))
    assert res == TFHardIntersection()(box1, box2)


def test_intersection_with_broadcasting_module() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]])
    )  # box_shape (2,2)
    box2 = TFBoxTensor(tf.Variable([[2, 0], [6, 2]]))  # box_shape (2,)
    res = TFBoxTensor(tf.Variable([[[2, 1], [3, 2]], [[2, 1], [3, 2]]]))
    assert res == TFHardIntersection()(box1, box2)
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]])
    )  # box_shape (2,2)
    box2 = TFBoxTensor(tf.Variable([[2, 0], [6, 2]]))  # box_shape (2,)
    assert res == TFHardIntersection()(box2, box1)


def test_intersection_with_broadcasting_module2() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[[1, 1], [4, 4]], [[2, 2], [5, 5]]]])
    )  # box_shape (1, 2,2)
    assert box1.box_shape == (1, 2, 2)
    box2 = TFBoxTensor(tf.Variable([[[[3, 3], [7, 6]]], [[[1, 3], [3, 4]]]]))
    assert box2.box_shape == (2, 1, 2)
    expected = TFBoxTensor(
        tf.Variable(
            [
                [[[3, 3], [4, 4]], [[3, 3], [5, 5]]],
                [[[1, 3], [3, 4]], [[2, 3], [3, 4]]],
            ]
        )
    )
    assert expected == TFHardIntersection()(box1, box2)
    box1 = TFBoxTensor(
        tf.Variable([[[[1, 1], [4, 4]], [[2, 2], [5, 5]]]])
    )  # box_shape (1, 2,2)
    assert box1.box_shape == (1, 2, 2)
    box2 = TFBoxTensor(tf.Variable([[[[3, 3], [7, 6]]], [[[1, 3], [3, 4]]]]))
    assert box2.box_shape == (2, 1, 2)
    expected = TFBoxTensor(
        tf.Variable(
            [
                [[[3, 3], [4, 4]], [[3, 3], [5, 5]]],
                [[[1, 3], [3, 4]], [[2, 3], [3, 4]]],
            ]
        )
    )
    assert expected == TFHardIntersection()(box2, box1)
