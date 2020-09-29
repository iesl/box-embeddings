from box_embeddings.initializations.uniform_boxes import uniform_boxes
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
import numpy as np
import torch
import pytest


@hypothesis.given(
    dimensions=integers(1, 100),
    num_boxes=integers(5, 50),
    minimum=floats(-10.0, 10.0),
    max_delta=floats(2.0, 10.0),
    delta_min=floats(0.1, 0.99),
    delta_max_delta=floats(0.1, 0.99),
)
@hypothesis.settings(max_examples=100, verbosity=hypothesis.Verbosity.verbose)
def test_uniform_boxes(
    dimensions, num_boxes, minimum, max_delta, delta_min, delta_max_delta
):
    maximum = minimum + max_delta
    delta_max = delta_min + delta_max_delta
    uniform_boxes(
        dimensions, num_boxes, minimum, maximum, delta_min, delta_max
    )
