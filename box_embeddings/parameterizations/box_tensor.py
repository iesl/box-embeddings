"""Base class for creating a wrapper around the torch.Tensor to represent boxes

A BoxTensor contains single tensor which represents single or multiple boxes.

    note:
        Have to use composition instead of inheritance because currently it is not safe to interit from :class:`torch.Tensor` because creating an instance of such a class will always make it a leaf node. This works for :class:`torch.nn.Parameter` but won't work for a general BoxTensor. This most likely will change in the future as pytorch starts offical support for inheriting from a Tensor. Give this point some thought when this happens.

"""
import torch
from torch import Tensor
from abc import ABC
from typing import List, Tuple, Union, Dict, Any, Optional, Type, TypeVar


def _box_shape_ok(t: Tensor) -> bool:
    if len(t.shape) < 2:
        return False
    else:
        if t.size(-2) != 2:
            return False

        return True


def _shape_error_str(tensor_name, expected_shape, actual_shape):
    return "Shape of {} has to be {} but is {}".format(tensor_name,
                                                       expected_shape,
                                                       tuple(actual_shape))


# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(ABC):
    """Abstract base class defining the interface for BoxTensor.
    """

    def __init__(self, data: Tensor) -> None:
        """
            Arguments:
                data: Tensor of shape (..., zZ, num_dims). Here, zZ=2, where
                    the 0th dim is for bottom left corner and 1st dim is for
                    top right corner of the box
        """

        if _box_shape_ok(data):
            self.data = data
        else:
            raise ValueError(
                _shape_error_str('data', '(...,2,num_dims)', data.shape))
        super().__init__()

    def __repr__(self):
        return 'box_tensor_wrapper(' + self.data.__repr__() + ')'

    @property
    def z(self) -> Tensor:
        """Lower left coordinate as Tensor"""

        return self.data[..., 0, :]

    @property
    def Z(self) -> Tensor:
        """Top right coordinate as Tensor"""

        return self.data[..., 1, :]

    @property
    def centre(self) -> Tensor:
        """Centre coordinate as Tensor"""

        return (self.z + self.Z) / 2

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        """
        Creates a box by stacking z and Z along -2 dim.
        That is if z.shape == Z.shape == (..., num_dim),
        then the result would be box of shape (..., 2, num_dim)
        """

        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape))
        box_val: Tensor = torch.stack((z, Z), -2)

        return cls(box_val)

    @classmethod
    def from_split(cls: Type[TBoxTensor], t: Tensor,
                   dim: int = -1) -> TBoxTensor:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
            t: input
            dim: dimension to split on

        Returns:
            BoxTensor: output BoxTensor

        Raises:
            ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(
                    t.size(dim)))
        split_point = int(len_dim / 2)
        z = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point)), dtype=torch.int64, device=t.device))

        Z = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=t.device))

        return cls.from_zZ(z, Z)
