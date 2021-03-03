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
from copy import deepcopy

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
    """Base class defining the interface for BoxTensor."""

    w2z_ratio: int = 2  #: number of parameters required per dim

    def __init__(
        self,
        data: Union[Tensor, Tuple[Tensor, Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Arguments:
            data: Tensor of shape (..., zZ, num_dims). Here, zZ=2, where
                the 0th dim is for bottom left corner and 1st dim is for
                top right corner of the box
            *args: TODO
            **kwargs: TODO
        """
        self.data: Optional[Tensor] = None
        self._z: Optional[Tensor] = None
        self._Z: Optional[Tensor] = None

        self.reinit(data)

        super().__init__()

    def __repr__(self) -> str:
        if self.data is not None:
            return (
                f"{self.__class__.__name__}"
                f"({self.data.__repr__()})"  # type:ignore
            )
        else:
            return (
                f"{self.__class__.__name__}(z={self._z.__repr__()},"
                f"\nZ={self._Z.__repr__()})"  # type:ignore
            )

    def reinit(self, data: Union[Tensor, Tuple[Tensor, Tensor]]) -> None:
        assert data is not None

        if self.data is not None:
            if isinstance(data, Tensor):
                if data.shape != self.data.shape:
                    raise ValueError(
                        f"Cannot reinitialize with different shape. New {data.shape}, old {self.data.shape}"
                    )

        if self._z is not None:
            if self._z.shape != data[0].shape:
                raise ValueError(
                    f"Cannot reinitialize with different shape. New z shape ={data[0].shape}, old ={self._z.shape}"
                )

        if self._Z is not None:
            if self._Z.shape != data[1].shape:
                raise ValueError(
                    f"Cannot reinitialize with different shape. New Z shape ={data[1].shape}, old ={self._Z.shape}"
                )

        if isinstance(data, Tensor):
            if _box_shape_ok(data):
                self.data = data
            else:
                raise ValueError(
                    _shape_error_str("data", "(...,2,num_dims)", data.shape)
                )
        else:
            self._z, self._Z = data

    @property
    def kwargs(self) -> Dict:
        """Configuration attribute values

        Returns:
            Dict
        """

        return {}

    @property
    def args(self) -> Tuple:
        return tuple()

    @property
    def z(self) -> Tensor:
        """Lower left coordinate as Tensor

        Returns:
            Tensor: lower left corner
        """

        if self.data is not None:
            return self.data[..., 0, :]
        else:
            return self._z  # type:ignore

    @property
    def Z(self) -> Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        if self.data is not None:
            return self.data[..., 1, :]
        else:
            return self._Z  # type:ignore

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
    def W(
        cls: Type[TBoxTensor], z: Tensor, Z: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        For the base `BoxTensor` class, we just return z and Z stacked together.
        If you implement any new parameterization for boxes. You most likely
        need to override this method.

        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)
            *args: TODO
            **kwargs: TODO

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).
        """
        cls.check_if_valid_zZ(z, Z)

        return torch.stack((z, Z), -2)

    @classmethod
    def zZ_to_embedding(
        cls, z: Tensor, Z: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        W = cls.W(z, Z, *args, **kwargs)
        # collapse the last two dimensions

        return W.reshape(*W.shape[:-2], -1)

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
        assert z.shape == Z.shape, "z,Z shape not matching"

        return cls((z, Z), *args, **kwargs)

    def like_this_from_zZ(
        self,
        z: Tensor,
        Z: Tensor,
    ) -> "BoxTensor":
        """Creates a box for the given min-max coordinates (z,Z).
        This is similar to the class method :method:`from_zZ`, but
        uses the attributes on self and not external args, kwargs.

        For the base class, since we do not have extra attributes,
        we simply call from_zZ.


        Args:
            z: lower left
            Z: top right

        Returns:
            A BoxTensor

        """

        return self.from_zZ(z, Z, *self.args, *self.kwargs)

    @classmethod
    def from_vector(
        cls, vector: Tensor, *args: Any, **kwargs: Any
    ) -> TBoxTensor:
        """Creates a box for a vector. In this base implementation the vector is split
        into two pieces and these are used as z,Z.

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
        z = vector.index_select(
            dim,
            torch.tensor(
                list(range(split_point)),
                dtype=torch.int64,
                device=vector.device,
            ),
        )

        Z = vector.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=vector.device,
            ),
        )

        return cls.from_zZ(z, Z, *args, **kwargs)  # type:ignore

    @property
    def box_shape(self) -> Tuple:
        """Shape of z, Z and center.

        Returns:
            Shape of z, Z and center.

        Note:
            This is *not* the shape of the `data` attribute.
        """

        if self.data is not None:
            data_shape = list(self.data.shape)
            _ = data_shape.pop(-2)

            return tuple(data_shape)
        else:
            assert self._z.shape == self._Z.shape  # type:ignore

            return self._z.shape  # type: ignore

    def broadcast(self, target_shape: Tuple) -> None:
        """Broadcasts the internal data member in-place such that z and Z
        return tensors that can be automatically broadcasted to perform
        arithmetic operations with shape `target_shape`.

        Ex:
            target_shape = (4,5,10)

            1. self.box_shape = (10,) => (1,1,10)
            2. self.box_shape = (3,) => ValueError
            3. self.box_shape = (4,10) => (4,1,10)
            4. self.box_shape = (4,2,10) =>  ValueError
            5. self.box_shape = (5,10) => (1,5,10)

        Note:
            This operation will not result in self.z, self.Z and self.center returning
            tensor of shape `target_shape` but it will result in return a tensor
            which is arithmetic compatible with `target_shape`.

        Args:
            target_shape: Shape of the broadcast target. Usually will be the shape of
                the tensor you wish to use z, Z with. For instance, if you wish to
                add self box's center [shape=(batch, hidden_dim)] with other
                box whose center's shape is (batch, extra_dim, hidden_dim), then
                this function will reshape the data such that the resulting center
                has shape (batch, 1, hidden_dim).

        Raises:
            ValueError: If bad target

        ..todo::
            Add an extra argument `repeat` which tell the
            function to repeat values till target is satisfied.
            This is needed for gumbel_intersection, where the broadcasted
            tensors need to be stacked.
        """
        self_box_shape = self.box_shape

        if self_box_shape[-1] != target_shape[-1]:
            raise ValueError(
                f"Cannot broadcast box of box_shape {self_box_shape} to {target_shape}."
                "Last dimensions should match."
            )

        if len(self_box_shape) > len(target_shape):
            # see if we have 1s in the right places in the self.box_shape
            raise ValueError(
                f"Lenght of self.box_shape ({len(self_box_shape)})"
                f" should be <= length of target_shape ({len(self_box_shape)})"
            )

        elif len(self_box_shape) == len(target_shape):
            # they can be of the form (4,1,10) & (1,4,10)

            for s_d, t_d in zip(self_box_shape[:-1], target_shape[:-1]):

                if not ((s_d == t_d) or (s_d == 1) or (t_d == 1)):
                    raise ValueError(
                        f"Incompatible shapes {self_box_shape} and target {target_shape}"
                    )
        else:  # <= target_shape
            potential_final_shape = list(self_box_shape)
            dim_to_unsqueeze = []

            for dim in range(-2, -len(target_shape) - 1, -1):  # (-2, -3, ...)
                if (
                    dim + len(potential_final_shape) < 0
                ):  # self has more dims left
                    potential_final_shape.insert(dim + 1, 1)
                    # +1 because
                    # insert indexing in list
                    # works differently than unsqueeze
                    dim_to_unsqueeze.append(dim)

                    continue

                if potential_final_shape[dim] != target_shape[dim]:
                    potential_final_shape.insert(dim + 1, 1)
                    dim_to_unsqueeze.append(dim)

            # final check
            assert len(potential_final_shape) == len(target_shape)

            for p_d, t_d in zip(potential_final_shape, target_shape):
                if not ((p_d == 1) or (p_d == t_d)):
                    raise ValueError(
                        f"Cannot make box_shape {self_box_shape} compatible to {target_shape}"
                    )

            if self.data is not None:
                for d in dim_to_unsqueeze:
                    self.data.unsqueeze_(
                        d - 1
                    )  # -1 because of extra 2 at dim -2
            else:
                for d in dim_to_unsqueeze:
                    self._z.unsqueeze_(d)  # type:ignore
                    self._Z.unsqueeze_(d)  # type: ignore
            assert self.box_shape == tuple(potential_final_shape)

    def box_reshape(self, target_shape: Tuple) -> "BoxTensor":
        """Reshape the z,Z and center.

        Ex:
            1. self.box_shape = (5,10), target_shape = (-1,10), creates box_shape (5,10)
            2. self.box_shape = (5,4,10), target_shape = (-1,10), creates box_shape (20,10)
            4. self.box_shape = (20,10), target_shape = (10,2,10), creates box_shape (10,2,10)
            3. self.box_shape = (5,), target_shape = (-1,10),  raises RuntimeError
            5. self.box_shape = (5,10), target_shape = (2,10),  raises RuntimeError

        Args:
            target_shape: TODO

        Returns:
            TBoxTensor

        Raises:
            RuntimeError: If space dimensions, ie. the last dimensions do not match.
            RuntimeError: If cannot reshape the extra dimensions and torch.reshape raises.
        """

        if self.box_shape[-1] != target_shape[-1]:
            raise RuntimeError(
                "Cannot do box reshape if space dimensions do not match."
                f" Current space dim is {self.box_shape[-1]} and target is {target_shape[-1]}"
            )
        _target_shape = list(target_shape)

        if self.data is not None:
            _target_shape.insert(-1, 2)  # insert the zZ
            try:
                new_data = self.data.reshape(*_target_shape).contiguous()
            except RuntimeError as re:
                raise RuntimeError(
                    f"Cannot reshape current box_shape {self.box_shape} to {target_shape}"
                ) from re
            reshaped_box = self.__class__(new_data, *self.args, **self.kwargs)
        else:
            try:
                new_z = self._z.reshape(  # type:ignore
                    *_target_shape
                ).contiguous()
                new_Z = self._Z.reshape(  # type:ignore
                    *_target_shape
                ).contiguous()
            except RuntimeError as re:
                raise RuntimeError(
                    f"Cannot reshape current box_shape {self.box_shape} to {target_shape}"
                ) from re
            reshaped_box = self.like_this_from_zZ(new_z, new_Z)

        return reshaped_box

    def __eq__(self, other: TBoxTensor) -> bool:  # type:ignore
        return torch.allclose(self.z, other.z) and torch.allclose(
            self.Z, other.Z
        )

    def __getitem__(self, indx):
        """Creates a TBoxTensor for the min-max coordinates at the given indexes

        Args:
            indx: Indexes of the required boxes

        Returns:
            TBoxTensor
        """
        z_ = self.z[indx]
        Z_ = self.Z[indx]

        return self.from_zZ(z_, Z_)


class BoxFactory(Registrable):

    """A factory class which will be subclassed(one for each box type)."""

    box_registry: Dict[str, Tuple[Type[BoxTensor], Optional[str]]] = {}

    def __init__(self, name: str, kwargs_dict: Dict = None):
        self.name = name  #: Name of the registered BoxTensor class
        self.kwargs_dict = kwargs_dict or {}
        try:
            self.box_subclass, box_constructor = self.box_registry[name]
        except KeyError as ke:
            raise KeyError(
                f"{name} not present in box_registry: {list(self.box_registry.keys())}"
            )

        if not box_constructor:
            self.creator: Type[BoxTensor] = self.box_subclass
        else:
            try:
                self.creator = getattr(self.box_subclass, box_constructor)
            except AttributeError as ae:
                raise ValueError(
                    f"{self.box_subclass.__name__} registered as {name} "
                    f"with constructor {box_constructor} "
                    f"but no method {box_constructor} found."
                )

    @classmethod
    def register_box_class(
        cls,
        name: str,
        constructor: str = None,
        exist_ok: bool = False,
    ) -> Callable[[Type[BoxTensor]], Type[BoxTensor]]:
        """This is different from allennlp registrable because what this class registers
        is not subclasses but subclasses of BoxTensor

        Args:
            name: TODO
            constructor: TODO
            exist_ok: TODO

        Returns:
            ()

        """

        def add_box_class(subclass: Type[BoxTensor]) -> Type[BoxTensor]:
            if name in cls.box_registry:
                if exist_ok:
                    message = (
                        f"{name} has already been registered as a "
                        f"box class {cls.box_registry[name][0].__name__}"
                        f", but exist_ok=True, so overriting with {subclass.__name__}"
                    )
                    logger.warning(message)
                else:
                    message = (
                        f"Cannot register {name} as box class"
                        f"name already in use for {cls.box_registry[name][0].__name__}"
                    )
                    raise RuntimeError(message)
            cls.box_registry[name] = (subclass, constructor)

            return subclass

        return add_box_class

    def __call__(self, *args: Any, **kwargs: Any) -> BoxTensor:

        return self.creator(*args, **kwargs, **self.kwargs_dict)  # type: ignore


BoxFactory.register("box_factory")(BoxFactory)  # register itself

BoxFactory.register_box_class("boxtensor")(BoxTensor)
BoxFactory.register_box_class("boxtensor_from_zZ", "from_zZ")(BoxTensor)
BoxFactory.register_box_class("boxtensor_from_vector", "from_vector")(
    BoxTensor
)
