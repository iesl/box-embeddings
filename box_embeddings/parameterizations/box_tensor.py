"""Base class for creating a wrapper around the torch.Tensor to represent boxes

A BoxTensor contains single tensor which represents single or multiple boxes.

    note:
        Have to use composition instead of inheritance because currently it is not safe to interit from :class:`torch.Tensor` because creating an instance of such a class will always make it a leaf node. This works for :class:`torch.nn.Parameter` but won't work for a general BoxTensor. This most likely will change in the future as pytorch starts offical support for inheriting from a Tensor. Give this point some thought when this happens.

"""
import torch
from torch import Tensor
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Type,
    TypeVar,
    Callable,
)
from box_embeddings.common.registrable import Registrable
import logging

logger = logging.getLogger(__name__)


def _box_shape_ok(t: Tensor) -> bool:
    if len(t.shape) < 2:
        return False
    else:
        if t.size(-2) != 2:
            return False

        return True


def _shape_error_str(
    tensor_name: str, expected_shape: Any, actual_shape: Tuple
) -> str:
    return "Shape of {} has to be {} but is {}".format(
        tensor_name, expected_shape, tuple(actual_shape)
    )


# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(object):
    """Base class defining the interface for BoxTensor.
    """

    def __init__(self, data: Tensor) -> None:
        """
        Constructor.

        Arguments:
            data: Tensor of shape (..., zZ, num_dims). Here, zZ=2, where
                the 0th dim is for bottom left corner and 1st dim is for
                top right corner of the box

        Raises:
            ValueError: If shape of `data` is not correct.
        """

        if _box_shape_ok(data):
            self.data = data
        else:
            raise ValueError(
                _shape_error_str("data", "(...,2,num_dims)", data.shape)
            )
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.data.__repr__()})"  # type:ignore
        )

    @property
    def z(self) -> Tensor:
        """Lower left coordinate as Tensor

        Returns:
            Tensor: lower left corner
        """

        return self.data[..., 0, :]

    @property
    def Z(self) -> Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        return self.data[..., 1, :]

    @property
    def centre(self) -> Tensor:
        """Centre coordinate as Tensor

        Returns:
            Tensor: Center
        """

        return (self.z + self.Z) / 2

    @classmethod
    def check_if_valid_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> None:
        """Check of (z,Z) form a valid box.

        If your child class parameterization bounds the boxes to some universe
        box then this is the right place to check that.

        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)

        Raises:
            ValueError: If `z` and `Z` do not have the same shape
            ValueError: If `Z` < `z`

        """

        if not (Z >= z).all().item():  # type: ignore
            raise ValueError(f"Invalid box: Z < z where\nZ = {Z}\nz={z}")

        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape
                )
            )

    @classmethod
    def W(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        For the base `BoxTensor` class, we just return z and Z stacked together.
        If you implement any new parameterization for boxes. You most likely
        need to override this method.

        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).
        """

        return torch.stack((z, Z), -2)

    @classmethod
    def from_zZ(
        cls: Type[TBoxTensor], z: Tensor, Z: Tensor, *args: Any, **kwargs: Any
    ) -> TBoxTensor:
        """Creates a box for the given min-max coordinates (z,Z).

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
        cls.check_if_valid_zZ(z, Z)

        return cls(cls.W(z, Z))


R = TypeVar("R", bound="BoxTensor")


class BoxFactory(Registrable):

    """A factory class which will be subclassed(one for each box type)."""

    box_registry: Dict[str, Tuple[Type[R], str]] = {}  # type:ignore

    def __init__(self, name: str, kwargs_dict: Dict = None):
        self.name = str  #: Name of the registered BoxTensor class
        self.kwargs_dict = kwargs_dict
        try:
            box_subclass, box_constructor = self.box_registry[name]
        except KeyError as ke:
            raise KeyError(
                f"{name} not present in box_registry: {list(self.box_registry.keys())}"
            )

        if not box_constructor:
            self.creator: Type[TBoxTensor] = box_subclass  # type: ignore
        else:
            try:
                self.creator = getattr(box_subclass, box_constructor)
            except AttributeError as ae:
                raise ValueError(
                    f"{box_subclass.__name__} registered as {name} "
                    f"with constructor {box_constructor} "
                    f"but no method {box_constructor} found."
                )

    @classmethod
    def register_box_class(
        cls, name: str, constructor: str = None, exist_ok: bool = False,
    ) -> Callable[[Type[BoxTensor]], Type[BoxTensor]]:
        """This is different from allennlp registrable because what this class registers
        is not subclasses but subclasses of BoxTensor

        Args:
            name: TODO
            constructor: TODO
            exist_ok: TODO

        Returns:
            ()

        Raises:
            RuntimeError: if

        """

        def add_box_class(subclass: Type[TBoxTensor]) -> Type[TBoxTensor]:
            if name in cls.box_registry:
                if exist_ok:
                    message = (
                        f"{name} has already been registered as a "  # type:ignore
                        f"box class {cls.box_registry[name][0].__name__}"
                        f", but exist_ok=True, so overriting with {subclass.__name__}"
                    )
                    logger.warning(message)
                else:
                    message = (
                        f"Cannot register {name} as box class"  # type:ignore
                        f"name already in use for {cls.box_registry[name][0].__name__}"
                    )
                    raise RuntimeError(message)
            cls.box_registry[name] = (subclass, constructor)  # type: ignore

            return subclass

        return add_box_class

    def __call__(self, *args: Any, **kwargs: Any) -> BoxTensor:

        return self.creator(*args, **kwargs)  # type:ignore


BoxFactory.register_box_class("boxtensor")(BoxTensor)
BoxFactory.register_box_class("boxtensor_from_zZ", "from_zZ")(BoxTensor)
