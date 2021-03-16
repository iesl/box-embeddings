from box_embeddings.common.tf_utils import logsumexp2
import tensorflow as tf
import numpy as np
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, booleans, sampled_from

import pytest


@hypothesis.given(
    inp1=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-10, 10),
    ),
    inp2=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-10, 10),
    ),
)
@hypothesis.settings(
    max_examples=500,
    # verbosity=hypothesis.Verbosity.verbose
)
def test_logsumexp2(inp1, inp2):
    """Checks that logsumexp2 does the same thing as
    torch.logsumexp and produces same gradients as well.
    """
    t1_ = tf.constant(inp1)
    t2_ = tf.constant(inp2)
    expected = tf.reduce_logsumexp(tf.stack((t1_, t2_)), axis=0)
    t1 = tf.constant(inp1)
    t2 = tf.constant(inp2)
    result = logsumexp2(t1, t2)
    assert np.allclose(result, expected)


@hypothesis.given(
    inp1=arrays(
        shape=(3, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-1000, 1000),
    ),
    inp2=arrays(
        shape=(3, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-1000, 1000),
    ),
)
@hypothesis.settings(
    max_examples=500,
    # verbosity=hypothesis.Verbosity.verbose
)
def test_logsumexp2_numerical_stability(inp1, inp2):
    """Checks that logsumexp2 does the same thing as
    torch.logsumexp and produces same gradients as well.
    """
    t1_ = tf.constant(inp1)
    t2_ = tf.constant(inp2)
    expected = tf.reduce_logsumexp(tf.stack((t1_, t2_)), axis=0)
    t1 = tf.constant(inp1)
    t2 = tf.constant(inp2)
    result = logsumexp2(t1, t2)
    # test value
    assert np.allclose(result, expected)
    # test bound w.r.t max
    t1__ = tf.constant(inp1)
    t2__ = tf.constant(inp2)
    max_ = tf.math.maximum(t1__, t2__)
    assert tf.reduce_all(max_ <= result)


@hypothesis.given(
    sample=sampled_from(
        [
            ((4, 5, 10), (1, 5, 10), (4, 5, 10)),
            ((4, 1, 10), (1, 5, 10), (4, 5, 10)),
            ((4, 5, 10), (4, 2, 10), tf.errors.InvalidArgumentError),
            ((4, 5, 10), (10,), (4, 5, 10)),
            ((4, 5, 10), (5,), tf.errors.InvalidArgumentError),
        ]
    )
)
def test_logsumexp2_manual_broadcasting(sample):
    t1_shape, t2_shape, res_shape = sample

    if res_shape == tf.errors.InvalidArgumentError:
        with pytest.raises(res_shape):
            res = logsumexp2(
                tf.constant(np.random.rand(*t1_shape)),
                tf.constant(np.random.rand(*t2_shape)),
            )
    else:
        res = logsumexp2(
            tf.constant(np.random.rand(*t1_shape)),
            tf.constant(np.random.rand(*t2_shape)),
        )
        assert res.shape == res_shape
