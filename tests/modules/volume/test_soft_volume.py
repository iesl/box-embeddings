import torch
from box_embeddings.modules.volume.volume import HardVolume
from box_embeddings.modules.volume.soft_volume import SoftVolume, eps
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.common.testing.test_case import BaseTestCase
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, booleans
import numpy as np


class TestSoftVolume(BaseTestCase):
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
        beta=floats(1.0, 50.0),
        log_scale=booleans(),
    )
    @hypothesis.settings(
        max_examples=100, verbosity=hypothesis.Verbosity.verbose
    )
    def test_volume(
        self, inp1: np.ndarray, inp2: np.ndarray, beta: float, log_scale: bool
    ) -> None:
        inp1[..., 1] = (
            np.absolute(inp1[..., 1]) + inp1[..., 0]
        )  # make sure Z >z
        inp2[..., 1] = (
            np.absolute(inp2[..., 1]) + inp2[..., 0]
        )  # make sure Z >z
        box1 = BoxTensor(torch.tensor(inp1))
        box2 = BoxTensor(torch.tensor(inp2))
        hard_volume1 = HardVolume(log_scale=log_scale, beta=beta)(box1)
        soft_volume1 = SoftVolume(log_scale=log_scale, beta=beta)(box1)


#        if not log_scale:
#            tol = (
#                torch.pow(
#                    torch.nn.functional.softplus(torch.tensor(0.0), beta=beta),
#                    inp1.shape[-1],
#                ).item() +
#                1e-4
#            )
#        else:
#            tol = (
#                inp1.shape[-1] *
#                (
#                    torch.abs(
#                        torch.log(
#                            torch.nn.functional.softplus(
#                                torch.tensor(0.0), beta=beta
#                            ).clamp_min(eps)
#                        ) -
#                        torch.log(torch.tensor(eps))
#                    )
#                ) +
#                1e-4
#            )
#
#        if not log_scale:
#            assert (soft_volume1 >= 0).all(), "soft volume greater than zero"
#        assert torch.allclose(soft_volume1, hard_volume1, atol=tol), ""
