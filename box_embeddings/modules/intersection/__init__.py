import logging
from ... import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .intersection import Intersection
    from .hard_intersection import hard_intersection, HardIntersection
    from .gumbel_intersection import gumbel_intersection, GumbelIntersection

if tensorflow_is_available():
    from .tf_intersection import TFIntersection
    from .tf_hard_intersection import tf_hard_intersection, TFHardIntersection

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
