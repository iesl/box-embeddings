import logging
from .. import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .initializer import BoxInitializer
    from .uniform_boxes import UniformBoxInitializer
    from .uniform_boxes import uniform_boxes

if tensorflow_is_available():
    from .tf_initializer import TFBoxInitializer
    from .tf_uniform_boxes import TFUniformBoxInitializer
    from .tf_uniform_boxes import tf_uniform_boxes

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
