import logging
from ... import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .pooling import BoxPooler
    from .intersection import HardIntersectionBoxPooler
    from .bag_of_boxes import bag_of_boxes_pooler, BagOfBoxesBoxPooler

if tensorflow_is_available():
    from .tf_pooling import TFBoxPooler
    from .tf_intersection import TFHardIntersectionBoxPooler
    from .tf_bag_of_boxes import tf_bag_of_boxes_pooler, TFBagOfBoxesBoxPooler

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
