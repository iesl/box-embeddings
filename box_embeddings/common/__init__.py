import logging
from .. import torch_is_available, tensorflow_is_available

ALLENNLP_PRESENT = True
try:
    import allennlp
except ImportError as ie:
    ALLENNLP_PRESENT = False

logger = logging.getLogger(__name__)

if torch_is_available():
    from .utils import (
        tiny_value_of_dtype,
        log1mexp,
        log1pexp,
        softplus_inverse,
        logsumexp2,
        inv_sigmoid,
    )

elif tensorflow_is_available():
    from .tf_utils import (
        tiny_value_of_dtype,
        log1mexp,
        log1pexp,
        softplus_inverse,
        logsumexp2,
        inv_sigmoid,
        tf_index_select,
        _box_shape_ok,
        _shape_error_str,
    )

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
