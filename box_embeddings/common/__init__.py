import logging
from .. import torch_is_available, tensorflow_is_available
from .registrable import DummyRegistrable

logger = logging.getLogger(__name__)

if torch_is_available():
    from .utils import *

if tensorflow_is_available():
    from .tf_utils import *

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")

ALLENNLP_PRESENT = True
try:
    import allennlp
except ImportError as ie:
    ALLENNLP_PRESENT = False
