import logging
from ... import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .regularizer import BoxRegularizer
    from .l2_side_regularizer import l2_side_regularizer, L2SideBoxRegularizer

if tensorflow_is_available():
    from .tf_regularizer import TFBoxRegularizer
    from .tf_l2_side_regularizer import (
        tf_l2_side_regularizer,
        TFL2SideBoxRegularizer,
    )

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
