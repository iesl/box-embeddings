from box_embeddings.initializations.tf_uniform_boxes import (
    tf_uniform_boxes,
    TFUniformBoxInitializer,
)
from box_embeddings.parameterizations.tf_box_tensor import TFBoxFactory
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, sampled_from
import numpy as np
import tensorflow as tf
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
    tf_uniform_boxes(
        dimensions, num_boxes, minimum, maximum, delta_min, delta_max
    )


@hypothesis.given(
    dimensions=integers(1, 100),
    num_boxes=integers(5, 50),
    minimum=floats(-10.0, 10.0),
    max_delta=floats(2.0, 10.0),
    delta_min=floats(0.1, 0.99),
    delta_max_delta=floats(0.1, 0.99),
    box_type=sampled_from(["mindelta", "boxtensor"]),
)
@hypothesis.settings(max_examples=100, verbosity=hypothesis.Verbosity.verbose)
def test_UniformBoxInitializer(
    dimensions,
    num_boxes,
    minimum,
    max_delta,
    delta_min,
    delta_max_delta,
    box_type,
):
    maximum = minimum + max_delta
    delta_max = delta_min + delta_max_delta
    box_type_factory = TFBoxFactory(box_type)
    t = tf.Variable(
        np.random.rand(
            num_boxes, box_type_factory.box_subclass.w2z_ratio, dimensions
        )
    )
    TFUniformBoxInitializer(
        dimensions,
        num_boxes,
        box_type_factory=box_type_factory,
        minimum=minimum,
        maximum=maximum,
        delta_min=delta_min,
        delta_max=delta_max,
    )(t)
    b = box_type_factory(t)
    assert tf.math.reduce_all(b.Z <= maximum)
    assert tf.math.reduce_all(b.z >= minimum)
