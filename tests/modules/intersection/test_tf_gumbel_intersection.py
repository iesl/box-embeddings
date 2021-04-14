from box_embeddings.parameterizations import (
    TFBoxTensor,
    TFMinDeltaBoxTensor,
)
from box_embeddings.modules.intersection.tf_hard_intersection import (
    tf_hard_intersection,
    TFHardIntersection,
)
from box_embeddings.modules.intersection import (
    TFGumbelIntersection,
    tf_gumbel_intersection,
)

'''
from box_embeddings.modules.volume.bessel_volume import (
    log_bessel_volume_approx,
)
'''
import tensorflow as tf
import numpy as np
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, booleans, sampled_from
from box_embeddings.common.tf_utils import log1mexp
import pytest


# torch.autograd.set_detect_anomaly(True)


def test_intersection_with_fixed_input() -> None:
    box1 = TFBoxTensor(
        tf.Variable([[[1, 1], [3, 5]], [[1, 1], [3, 3]]], dtype=tf.float64)
    )
    box2 = TFBoxTensor(
        tf.Variable([[[2, 0], [6, 2]], [[3, 2], [4, 4]]], dtype=tf.float64)
    )
    hard_res = TFBoxTensor(
        tf.Variable([[[2, 1], [3, 2]], [[3, 2], [3, 3]]], dtype=tf.float64)
    )
    tf_gumbel_intersection(box1, box2)


@hypothesis.given(
    inp1=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    inp2=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    beta=floats(1e-5, 1.0),
    approximation_mode=sampled_from(["clipping", "clipping_forward"]),
    box_type=sampled_from([TFMinDeltaBoxTensor, TFBoxTensor]),
)
@hypothesis.settings(print_blob=True, max_examples=100)
def test_intersection_all_input_ranges(
    inp1, inp2, beta, approximation_mode, box_type
) -> None:
    box1 = box_type(tf.Variable(inp1, dtype=tf.float64))
    box2 = box_type(tf.Variable(inp2, dtype=tf.float64))
    res = tf_gumbel_intersection(
        box1, box2, gumbel_beta=beta, approximation_mode=approximation_mode
    )
    assert tf.reduce_all(tf.math.is_finite(res.z))
    assert tf.reduce_all(tf.math.is_finite(res.Z))
    hard_res = tf_hard_intersection(box1, box2)
    assert tf.reduce_all(res.z >= hard_res.z)
    assert tf.reduce_all(res.Z <= hard_res.Z)


@hypothesis.given(
    inp1=arrays(
        shape=(3, 1, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),  # box_shape (3,1,20)
    inp2=arrays(
        shape=(1, 4, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),  # box_shape (1,4,10)
    beta=floats(1e-5, 1.0),
    approximation_mode=sampled_from(["clipping", "clipping_forward"]),
    box_type=sampled_from([TFMinDeltaBoxTensor, TFBoxTensor]),
)
@hypothesis.settings(
    max_examples=1000,
    print_blob=True
    # verbosity=hypothesis.Verbosity.verbose
)
def test_intersection_with_broadcasting(
    inp1, inp2, beta, approximation_mode, box_type
) -> None:
    box1 = box_type(tf.Variable(inp1))
    box2 = box_type(tf.Variable(inp2))
    hard_res = tf_hard_intersection(box1, box2)
    g1 = tf_gumbel_intersection(
        box1, box2, gumbel_beta=beta, approximation_mode=approximation_mode
    )
    g2 = tf_gumbel_intersection(
        box2, box1, gumbel_beta=beta, approximation_mode=approximation_mode
    )
    assert tf.reduce_all(hard_res.z <= g1.z)
    assert tf.reduce_all(hard_res.z <= g2.z)
    assert tf.reduce_all(hard_res.Z >= g1.Z)
    assert tf.reduce_all(hard_res.Z >= g2.Z)


@hypothesis.given(
    inp1=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    inp2=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    gumbel_beta=floats(1e-5, 1.0),
    beta=floats(1.0, 50.0),
    expected_probs=arrays(
        shape=(3,),
        dtype=np.float,
        elements=hypothesis.strategies.integers(0, 1),
    ),
    approximation_mode=sampled_from([None, "clipping", "clipping_forward"]),
    box_type=sampled_from([TFMinDeltaBoxTensor, TFBoxTensor]),
)
@hypothesis.settings(
    print_blob=True,
    max_examples=1000,
)
def test_intersection_all_input_ranges_grad_computation(
    inp1, inp2, gumbel_beta, beta, expected_probs, approximation_mode, box_type
) -> None:
    t1 = tf.Variable(inp1, dtype=tf.float64)
    t2 = tf.Variable(inp2, dtype=tf.float64)
    box1 = box_type(t1)
    box2 = box_type(t2)
    res = tf_gumbel_intersection(
        box1,
        box2,
        gumbel_beta=gumbel_beta,
        approximation_mode=approximation_mode,
    )
    assert tf.reduce_all(tf.math.is_finite(res.z))
    assert tf.reduce_all(tf.math.is_finite(res.Z))
    hard_res = tf_hard_intersection(box1, box2)

    if approximation_mode is not None:
        assert tf.reduce_all(res.z >= hard_res.z)
        assert tf.reduce_all(res.Z <= hard_res.Z)

    '''
    cp_1 = log_bessel_volume_approx(
        res, beta=beta, gumbel_beta=gumbel_beta
    ) - log_bessel_volume_approx(box1, beta=beta, gumbel_beta=gumbel_beta)
    cp_2 = log_bessel_volume_approx(
        res, beta=beta, gumbel_beta=gumbel_beta
    ) - log_bessel_volume_approx(box2, beta=beta, gumbel_beta=gumbel_beta)
    expected_probs = torch.tensor(expected_probs).long()
    loss1 = torch.nn.NLLLoss()(
        torch.cat((cp_1.unsqueeze(-1), log1mexp(cp_1.unsqueeze(-1))), dim=-1),
        expected_probs,
    )
    loss1.backward()

    assert tf.reduce_all(tf.math.is_finite(t1.grad))
    assert tf.reduce_all(tf.math.is_finite(t2.grad))
    '''


@pytest.mark.xfail
@hypothesis.given(
    inp1=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    inp2=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    gumbel_beta=floats(1e-5, 1.0),
    beta=floats(1.0, 50.0),
    box_type=sampled_from([TFMinDeltaBoxTensor, TFBoxTensor]),
    approximation_mode=sampled_from(["clipping", "clipping_forward"]),
)
@hypothesis.settings(
    print_blob=True,
    max_examples=1000,
)
def test_intersection_all_input_ranges_grad_value(
    inp1, inp2, gumbel_beta, beta, box_type, approximation_mode
) -> None:
    t1 = tf.Variable(inp1, dtype=tf.float64)
    t2 = tf.Variable(inp2, dtype=tf.float64)
    box1 = box_type(t1)
    box2 = box_type(t2)
    res1 = tf_gumbel_intersection(
        box1, box2, gumbel_beta=gumbel_beta, approximation_mode=None
    )
    '''
    l1 = torch.mean(
        log_bessel_volume_approx(res1, beta=beta, gumbel_beta=gumbel_beta)
    )
    t1_ = torch.tensor(inp1, dtype=torch.float, requires_grad=True)
    t2_ = torch.tensor(inp2, dtype=torch.float, requires_grad=True)
    box1_ = box_type(t1_)
    box2_ = box_type(t2_)
    res1_ = gumbel_intersection(
        box1_,
        box2_,
        gumbel_beta=gumbel_beta,
        approximation_mode=approximation_mode,
    )
    l1_ = torch.mean(
        log_bessel_volume_approx(res1_, beta=beta, gumbel_beta=gumbel_beta)
    )
    l1_.backward()
    l1.backward()
    assert np.allclose(t1.grad, t1_.grad, atol=1e-4)
    assert np.allclose(t2.grad, t2_.grad, atol=1e-4)
    '''


@hypothesis.given(
    inp1=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    inp2=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    beta=floats(1e-5, 1.0),
)
def test_intersection_module(inp1, inp2, beta) -> None:
    box1 = TFBoxTensor(tf.Variable(inp1))
    box2 = TFBoxTensor(tf.Variable(inp2))
    g1 = TFGumbelIntersection(beta=beta)(box1, box2)
    g2 = tf_gumbel_intersection(box1, box2, gumbel_beta=beta)
    assert np.allclose(g1.z, g2.z)
    assert np.allclose(g1.Z, g2.Z)
