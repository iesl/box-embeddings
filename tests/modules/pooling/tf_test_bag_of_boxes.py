from box_embeddings.modules.pooling.tf_bag_of_boxes import (
    TFBagOfBoxesBoxPooler,
)
from box_embeddings.parameterizations.tf_box_tensor import TFBoxTensor
import hypothesis
from hypothesis.strategies import sampled_from, just
import tensorflow as tf
import pytest
import numpy as np
from tensorflow.errors import InternalError


@hypothesis.given(
    others=sampled_from(
        [
            {
                "box": TFBoxTensor(
                    tf.Variable(
                        [[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float32
                    )
                ),
                "weights": None,
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": TFBoxTensor(
                    tf.Variable(
                        [[3.0 / 2.0, 1.0 / 2.0], [9.0 / 2.0, 7.0 / 2.0]]
                    )
                ),
            },
            {
                "box": TFBoxTensor(
                    tf.Variable(
                        [[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float32
                    )
                ),
                "weights": tf.reshape(tf.Variable([0.5, 0.5]), (2, 1)),
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": TFBoxTensor(
                    tf.Variable(
                        [[3.0 / 2.0, 1.0 / 2.0], [9.0 / 2.0, 7.0 / 2.0]]
                    )
                ),
            },
            {
                "box": TFBoxTensor(
                    tf.Variable(
                        [[[1, 1], [3, 5]], [[2, 0], [6, 2]]], dtype=tf.float32
                    )
                ),
                "weights": tf.reshape(tf.Variable([0.1, 0.9]), (2, 1)),
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": TFBoxTensor(
                    tf.Variable(
                        [
                            [(0.1 * 1.0 + 0.9 * 2.0), (0.1 * 1.0 + 0.9 * 0)],
                            [0.1 * 3 + 0.9 * 6.0, 0.1 * 5 + 0.9 * 2],
                        ]
                    )
                ),
            },
            {
                "box": TFBoxTensor(
                    tf.Variable(
                        [
                            [[[1, 1], [3, 5]], [[2, 0], [6, 2]]]
                        ],  # data shape (1,2,2,2); box shape (1,2,2)
                        dtype=tf.float32,
                    )
                ),
                "weights": tf.reshape(tf.Variable([0.1, 0.9]), (1, 2, 1)),
                "mask": None,
                "keepdim": True,
                "dim": 1,
                "expected": TFBoxTensor(
                    tf.Variable(
                        [
                            [
                                [
                                    (0.1 * 1.0 + 0.9 * 2.0),
                                    (0.1 * 1.0 + 0.9 * 0),
                                ],
                                [0.1 * 3 + 0.9 * 6.0, 0.1 * 5 + 0.9 * 2],
                            ]
                        ],
                        dtype=tf.float32,
                    )
                ),
            },
            {
                "box": TFBoxTensor(
                    tf.Variable(
                        [[[1, 1, 1], [3, 5, 6]], [[2, 0, 1], [6, 2, 3]]],
                        dtype=tf.float32,
                    )
                ),
                "weights": tf.Variable([0.5, 0.5], dtype=tf.float32),
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": InternalError,
            },
        ]
    ),
)
def test_bob(others) -> None:
    box = others["box"]
    bob = TFBagOfBoxesBoxPooler(dim=others["dim"], keepdim=others["keepdim"])
    expected = others["expected"]

    if isinstance(expected, TFBoxTensor):
        # if len(others["weights"].shape) == 1:
        #    breakpoint()
        result = bob(box, mask=others["mask"], weights=others["weights"])
        assert np.allclose(result.z, others["expected"].z)
        assert np.allclose(result.Z, others["expected"].Z)
    else:
        with pytest.raises(expected):
            result = bob(box, mask=others["mask"], weights=others["weights"])
