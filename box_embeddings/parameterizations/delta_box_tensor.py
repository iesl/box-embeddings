"""
    Implementation of min-delta box parameterization.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type
from box_embeddings.parameterizations.box_tensor import (
    BoxTensor,
    BoxFactory,
    TBoxTensor,
)
from box_embeddings.common.utils import softplus_inverse
import torch
import warnings


@BoxFactory.register_box_class("mindelta")
class MinDeltaBoxTensor(BoxTensor):

    """Unconstrained min-delta box tensor.

    For input of the shape (..., 2, box_dim), this parameterization
    defines z=w, and Z=w + delta, where w and delta come from the -2th dimension
    of the input. It uses softplus to keep the delta positive.

    """

    def __init__(
        self, data: torch.Tensor, beta: float = 1.0, threshold: float = 20
    ):
        """

        Args:
            data: The weights for the box
            beta: beta parameter for softplus for delta. Depending on the
                universe box and your inputs ranges, you might want to change this.
                Higher values of beta will make softplus harder and bring it close to ReLU.
            threshold: parameter for the softplus for delta
        """
        super().__init__(data)
        self.beta = beta
        self.threshold = threshold

    @property
    def z(self) -> torch.Tensor:
        """Lower left coordinate as Tensor

        Returns:
            torch.Tensor: lower left corner
        """

        return self.data[..., 0, :]

    @property
    def Z(self) -> torch.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        return self.z + torch.nn.functional.softplus(
            self.data[..., 1, :], beta=self.beta
        )

    @property
    def centre(self) -> torch.Tensor:
        """Centre coordinate as Tensor

        Returns:
            Tensor: Center
        """

        return (self.z + self.Z) / 2.0

    @classmethod
    def W(
        cls: Type[TBoxTensor], z: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        The min coordinate is stored as is:
        W[...,0,:] = z
        W[...,1,:] = softplus_inverse(Z-z)

        The max coordinate is transformed


        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).
        """
        cls.check_if_valid_zZ(z, Z)

        if ((Z - z) < 0).any():
            warnings.warn(
                "W() method for MinDeltaBoxTensor is numerically unstable."
                " It can produce high error when input Z-z is < 0."
            )

        return torch.stack((z, softplus_inverse(Z - z)), -2)

    @classmethod
    def from_zZ(  # type:ignore
        cls,
        z: torch.Tensor,
        Z: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20,
    ) -> BoxTensor:
        """Creates a box for the given min-max coordinates (z,Z).

        In the this base implementation we do this by
        stacking z and Z along -2 dim to form W.

        Args:
            z: lower left
            Z: top right
            beta: beta parameter for softplus for delta. Depending on the
                universe box and your inputs ranges, you might want to change this.
                Higher values of beta will make softplus harder and bring it close to ReLU.
            threshold: parameter for the softplus for delta


        Returns:
            A BoxTensor

        """
        cls.check_if_valid_zZ(z, Z)

        return cls(cls.W(z, Z))


BoxFactory.register_box_class("mindelta_from_zZ", "from_zZ")(MinDeltaBoxTensor)
