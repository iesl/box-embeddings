import logging
from .. import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .embedding import BoxEmbedding

if tensorflow_is_available():
    from .tf_embedding import TFBoxEmbedding

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
