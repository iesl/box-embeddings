from box_embeddings.parameterizations.tf_box_tensor import (
    TFBoxFactory,
    TFBoxTensor,
)
from box_embeddings.common.testing.test_case import BaseTestCase
from box_embeddings.common import ALLENNLP_PRESENT

import hypothesis
from hypothesis.extra.numpy import arrays
import tensorflow as tf
import numpy as np


class TestBoxFactory(BaseTestCase):
    @hypothesis.given(
        inp1=arrays(
            shape=(3, 2, 10),
            dtype=np.float,
            elements=hypothesis.strategies.floats(-100, 100),
        ),
        inp2=arrays(
            shape=(2, 10),
            dtype=np.float,
            elements=hypothesis.strategies.floats(-100, 100),
        ),
    )
    @hypothesis.settings(
        max_examples=5, verbosity=hypothesis.Verbosity.verbose
    )
    def test_simple_creation(self, inp1: np.ndarray, inp2: np.ndarray) -> None:
        inp1[..., 1] += inp1[..., 0]  # make sure Z>z
        tensor = tf.constant(inp1)
        box_tensor: TFBoxTensor = TFBoxFactory("boxtensor")(tensor)
        assert (tensor.numpy() == box_tensor.data.numpy()).all()  # type: ignore
        assert isinstance(box_tensor, TFBoxTensor)
        inp2[..., 1] += inp2[..., 0]
        tensor = tf.constant(inp2)
        box_tensor2: TFBoxTensor = TFBoxFactory("boxtensor")(tensor)
        assert (tensor.numpy() == box_tensor2.data.numpy()).all()  # type: ignore
        assert isinstance(box_tensor, TFBoxTensor)

    @hypothesis.given(
        z=arrays(
            shape=(3, 5, 10),
            dtype=np.float,
            elements=hypothesis.strategies.floats(-100, 100),
        ),
        Z=arrays(
            shape=(3, 5, 10),
            dtype=np.float,
            elements=hypothesis.strategies.floats(0, 100),
        ),
    )
    @hypothesis.settings(
        max_examples=5, verbosity=hypothesis.Verbosity.verbose
    )
    def test_creation_from_zZ(self, z, Z) -> None:
        box: TFBoxTensor = TFBoxFactory("boxtensor_from_zZ")(
            tf.constant(z), tf.constant(z + Z)
        )
        assert list(box.z.shape) == [3, 5, 10]
        assert type(box) is TFBoxTensor


if ALLENNLP_PRESENT:
    from allennlp.common.params import Params

    def test_box_factory_creation_from_params():
        params = Params(
            {
                "type": "box_factory",
                "name": "mindelta",
                "kwargs_dict": {"beta": 2.0},
            }
        )
        bf = TFBoxFactory.from_params(params)
