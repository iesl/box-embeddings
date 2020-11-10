from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.intersection import (
    GumbelIntersection,
    gumbel_intersection,
)
import torch
import numpy as np
from box_embeddings.modules.intersection.hard_intersection import (
    hard_intersection,
    HardIntersection,
)
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, booleans

import pytest


def test_intersection_with_fixed_input() -> None:
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [3, 3]]]).float()
    )
    box2 = BoxTensor(
        torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]).float()
    )
    hard_res = BoxTensor(
        torch.tensor([[[2, 1], [3, 2]], [[3, 2], [3, 3]]]).float()
    )
    gumbel_intersection(box1, box2)


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
@hypothesis.settings(print_blob=True)
def test_intersection_all_input_ranges(inp1, inp2, beta) -> None:
    box1 = BoxTensor(torch.tensor(inp1).float())
    box2 = BoxTensor(torch.tensor(inp2).float())
    res = gumbel_intersection(box1, box2, gumbel_beta=beta)
    assert torch.isfinite(res.z).all()
    assert torch.isfinite(res.Z).all()
    hard_res = hard_intersection(box1, box2)
    # breakpoint()
    assert (res.z >= hard_res.z).all()
    assert (res.Z <= hard_res.Z).all()


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
)
@hypothesis.settings(
    # max_examples=500,
    print_blob=True
    # verbosity=hypothesis.Verbosity.verbose
)
def test_intersection_with_broadcasting(inp1, inp2, beta) -> None:
    box1 = BoxTensor(torch.tensor(inp1))
    box2 = BoxTensor(torch.tensor(inp2))
    hard_res = hard_intersection(box1, box2)
    g1 = gumbel_intersection(box1, box2, gumbel_beta=beta)
    g2 = gumbel_intersection(box2, box1, gumbel_beta=beta)
    assert (hard_res.z <= g1.z).all()
    assert (hard_res.z <= g2.z).all()
    assert (hard_res.Z >= g1.Z).all()
    assert (hard_res.Z >= g2.Z).all()


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
    box1 = BoxTensor(torch.tensor(inp1))
    box2 = BoxTensor(torch.tensor(inp2))
    g1 = GumbelIntersection(beta=beta)(box1, box2)
    g2 = gumbel_intersection(box1, box2, gumbel_beta=beta)
    assert torch.allclose(g1.z, g2.z)
    # assert torch.allclose(g1.Z, g2.Z)
