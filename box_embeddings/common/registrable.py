"""
Ability to register classes. If AllenNLP is not available, then currently registering won't work.
"""
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    TypeVar,
    Type,
    Callable,
)
import logging
from box_embeddings.common import ALLENNLP_PRESENT

logger = logging.getLogger(__name__)

if ALLENNLP_PRESENT:
    from allennlp.common.registrable import Registrable

else:
    logger.warning("AllenNLP not available. Registrable won't work.")

T = TypeVar("T", bound="DummyRegistrable")


class DummyRegistrable(object):

    """Dummy class which implements 'registrable' method which does nothing."""

    @classmethod
    def register(
        cls: Type[T],
        name: str,
        constructor: str = None,
        exist_ok: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        """Transparent method

        Args:
            name : TODO
            constructor : TODO
            exist_ok : TODO

        Returns:
            a wrapped callable

        """

        def foo(subclass: Type[T]) -> Type[T]:
            return subclass

        return foo


if not ALLENNLP_PRESENT:
    Registrable = DummyRegistrable  # type:ignore # noqa
