from box_embeddings.modules.pooling.bag_of_boxes import BagOfBoxesBoxPooler
from box_embeddings.parameterizations.box_tensor import BoxTensor
import hypothesis
from hypothesis.strategies import sampled_from, just
import torch
import pytest


@hypothesis.given(
    others=sampled_from(
        [
            {
                "box": BoxTensor(
                    torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float()
                ),
                "weights": None,
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": BoxTensor(
                    torch.tensor(
                        [[3.0 / 2.0, 1.0 / 2.0], [9.0 / 2.0, 7.0 / 2.0]]
                    )
                ),
            },
            {
                "box": BoxTensor(
                    torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float()
                ),
                "weights": torch.tensor([0.5, 0.5]).reshape(2, 1),
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": BoxTensor(
                    torch.tensor(
                        [[3.0 / 2.0, 1.0 / 2.0], [9.0 / 2.0, 7.0 / 2.0]]
                    )
                ),
            },
            {
                "box": BoxTensor(
                    torch.tensor([[[1, 1], [3, 5]], [[2, 0], [6, 2]]]).float()
                ),
                "weights": torch.tensor([0.1, 0.9]).reshape(2, 1),
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": BoxTensor(
                    torch.tensor(
                        [
                            [(0.1 * 1.0 + 0.9 * 2.0), (0.1 * 1.0 + 0.9 * 0)],
                            [0.1 * 3 + 0.9 * 6.0, 0.1 * 5 + 0.9 * 2],
                        ]
                    )
                ),
            },
            {
                "box": BoxTensor(
                    torch.tensor(
                        [
                            [[[1, 1], [3, 5]], [[2, 0], [6, 2]]]
                        ]  # data shape (1,2,2,2); box shape (1,2,2)
                    ).float()
                ),
                "weights": torch.tensor([0.1, 0.9]).reshape(1, 2, 1),
                "mask": None,
                "keepdim": True,
                "dim": 1,
                "expected": BoxTensor(
                    torch.tensor(
                        [
                            [
                                [
                                    (0.1 * 1.0 + 0.9 * 2.0),
                                    (0.1 * 1.0 + 0.9 * 0),
                                ],
                                [0.1 * 3 + 0.9 * 6.0, 0.1 * 5 + 0.9 * 2],
                            ]
                        ]
                    )
                ),
            },
            {
                "box": BoxTensor(
                    torch.tensor(
                        [[[1, 1, 1], [3, 5, 6]], [[2, 0, 1], [6, 2, 3]]]
                    ).float()
                ),
                "weights": torch.tensor([0.5, 0.5]),
                "mask": None,
                "keepdim": True,
                "dim": 0,
                "expected": RuntimeError,
            },
        ]
    ),
)
def test_bob(others) -> None:
    box = others["box"]
    bob = BagOfBoxesBoxPooler(dim=others["dim"], keepdim=others["keepdim"])
    expected = others["expected"]

    if isinstance(expected, BoxTensor):
        # if len(others["weights"].shape) == 1:
        #    breakpoint()
        result = bob(box, mask=others["mask"], weights=others["weights"])
        assert torch.allclose(result.z, others["expected"].z)
        assert torch.allclose(result.Z, others["expected"].Z)
    else:
        with pytest.raises(expected):
            result = bob(box, mask=others["mask"], weights=others["weights"])
