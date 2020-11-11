from box_embeddings.common.utils import logsumexp2
import torch
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
    t1_ = torch.tensor(inp1, requires_grad=True)
    t2_ = torch.tensor(inp2, requires_grad=True)
    expected = torch.logsumexp(torch.stack((t1_, t2_)), dim=0)
    t1 = torch.tensor(inp1, requires_grad=True)
    t2 = torch.tensor(inp2, requires_grad=True)
    result = logsumexp2(t1, t2)
    with torch.no_grad():
        assert torch.allclose(result, expected)
    f_ = torch.sum(expected)
    f = torch.sum(result)
    f_.backward()
    f.backward()
    with torch.no_grad():
        assert torch.allclose(t1_.grad, t1.grad)
        assert torch.allclose(t2_.grad, t2.grad)


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
    t1_ = torch.tensor(inp1, requires_grad=True)
    t2_ = torch.tensor(inp2, requires_grad=True)
    expected = torch.logsumexp(torch.stack((t1_, t2_)), dim=0)
    t1 = torch.tensor(inp1, requires_grad=True)
    t2 = torch.tensor(inp2, requires_grad=True)
    result = logsumexp2(t1, t2)
    # test value
    with torch.no_grad():
        assert torch.allclose(result, expected)
    # test grad
    f_ = torch.sum(expected)
    f = torch.sum(result)
    f_.backward()
    f.backward()
    with torch.no_grad():
        assert torch.allclose(t1_.grad, t1.grad)
        assert torch.allclose(t2_.grad, t2.grad)
    # test bound w.r.t max
    t1__ = torch.tensor(inp1)
    t2__ = torch.tensor(inp2)
    max_ = torch.max(t1__, t2__)
    assert (max_ <= result).all()


@hypothesis.given(
    sample=sampled_from(
        [
            ((4, 5, 10), (1, 5, 10), (4, 5, 10)),
            ((4, 1, 10), (1, 5, 10), (4, 5, 10)),
            ((4, 5, 10), (4, 2, 10), RuntimeError),
            ((4, 5, 10), (10,), (4, 5, 10)),
            ((4, 5, 10), (5,), RuntimeError),
        ]
    )
)
def test_logsumexp2_manual_broadcasting(sample):
    t1_shape, t2_shape, res_shape = sample

    if res_shape == RuntimeError:
        with pytest.raises(res_shape):
            res = logsumexp2(torch.rand(*t1_shape), torch.rand(*t2_shape))
    else:
        res = logsumexp2(torch.rand(*t1_shape), torch.rand(*t2_shape))
        assert res.shape == res_shape
