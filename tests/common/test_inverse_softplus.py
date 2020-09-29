from box_embeddings.common.utils import softplus_inverse
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
import numpy as np
import torch
import pytest


@pytest.mark.xfail(reason="Stable inverse of softplus not implemented.")
@hypothesis.given(
    inp=arrays(
        shape=(3, 2, 10),
        dtype=np.float,
        elements=hypothesis.strategies.floats(-100, 100),
    ),
    beta=floats(1.0, 50.0),
    threshold=floats(20.0, 50.0),
)
@hypothesis.settings(max_examples=100, verbosity=hypothesis.Verbosity.verbose)
def test_softplus_inverse(inp, beta, threshold):

    if (inp.min()) < 0:
        tol = 1e-4
    else:
        tol = 1e-8

    inp = torch.tensor(inp)
    assert torch.allclose(
        softplus_inverse(
            torch.nn.functional.softplus(inp, beta=beta, threshold=threshold),
            beta=beta,
            threshold=threshold,
        ),
        inp,
        atol=tol,
    )
