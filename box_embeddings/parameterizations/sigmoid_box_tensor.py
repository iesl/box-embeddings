"""
    Implementation of sigmoid box parameterization.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type
from box_embeddings.parameterizations.box_tensor import (
    BoxTensor,
    BoxFactory,
    TBoxTensor,
)
from box_embeddings.common.utils import softplus_inverse, inv_sigmoid
import torch
import warnings


@BoxFactory.register_box_class("sigmoid")
class SigmoidBoxTensor(BoxTensor):

    """
    Sigmoid Box Tensor
    """

    def __init__(
        self, data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """

        Args:
            data: The weights for the box
        """
        super().__init__(data)

    @property
    def z(self) -> torch.Tensor:
        """Lower left coordinate as Tensor

        Returns:
            Tensor: lower left corner
        """

        if self.data is not None:
            return torch.sigmoid(self.data[..., 0, :])
        else:
            return self._z  # type:ignore

    @property
    def Z(self) -> torch.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        if self.data is not None:
            z = self.z
            return z + torch.sigmoid(self.data[..., 1, :]) * (1.0 - z)  # type: ignore
        else:
            return self._Z  # type:ignore

    @classmethod
    def W(  # type:ignore
        cls: Type[TBoxTensor],
        z: torch.Tensor,
        Z: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)
            *args: extra arguments for child class
            **kwargs: extra arguments for child class

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).
        """
        cls.check_if_valid_zZ(z, Z)

        eps = torch.finfo(z.dtype).tiny
        w1 = inv_sigmoid(z.clamp(eps, 1.0 - eps))
        w2 = inv_sigmoid(((Z - z) / (1.0 - z)).clamp(eps, 1.0 - eps))

        return torch.stack((w1, w2), -2)

    @classmethod
    def from_zZ(  # type:ignore
        cls: Type[TBoxTensor],
        z: torch.Tensor,
        Z: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> BoxTensor:
        """
        Creates a box for the given min-max coordinates (z,Z).

        In the this base implementation we do this by
        stacking z and Z along -2 dim to form W.

        Args:
            z: lower left
            Z: top right
            *args: extra arguments for child class
            **kwargs: extra arguments for child class

        Returns:
            A BoxTensor

        """

        W = cls.W(z, Z)
        return cls(W, args, kwargs)

    @classmethod
    def from_vector(  # type:ignore
        cls: Type[TBoxTensor], vector: torch.Tensor, *args: Any, **kwargs: Any
    ) -> BoxTensor:
        """Creates a box for a vector. In this base implementation the vector is split
        into two pieces and these are used as box weights.

        Args:
            vector: tensor
            *args: extra arguments for child class
            **kwargs: extra arguments for child class

        Returns:
            A BoxTensor

        Raises:
            ValueError: if last dimension is not even
        """
        len_dim = vector.shape[-1]
        dim = -1

        if vector.shape[-1] % 2 != 0:
            raise ValueError(
                f"The last dimension of vector should be even but is {vector.shape[-1]}"
            )

        split_point = int(len_dim / 2)
        w1 = vector.index_select(
            dim,
            torch.tensor(
                list(range(split_point)),
                dtype=torch.int64,
                device=vector.device,
            ),
        )

        w2 = vector.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=vector.device,
            ),
        )

        W: torch.Tensor = torch.stack((w1, w2), -2)

        return cls(W)


# BoxFactory.register_box_class("mindelta_from_zZ", "from_zZ")(MinDeltaBoxTensor)
BoxFactory.register_box_class("sigmoid_from_vector", "from_vector")(
    SigmoidBoxTensor
)
