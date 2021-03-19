import logging
from .. import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .box_tensor import TBoxTensor
    from .box_tensor import BoxTensor
    from .delta_box_tensor import MinDeltaBoxTensor
    from .sigmoid_box_tensor import SigmoidBoxTensor
    from .tanh_box_tensor import TanhBoxTensor

if tensorflow_is_available():
    from .tf_box_tensor import TFTBoxTensor
    from .tf_box_tensor import TFBoxTensor
    from .tf_delta_box_tensor import TFMinDeltaBoxTensor
    from .tf_sigmoid_box_tensor import TFSigmoidBoxTensor
    from .tf_tanh_box_tensor import TFTanhBoxTensor

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
