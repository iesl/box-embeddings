import logging
from ... import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .volume import Volume, hard_volume, log_hard_volume, HardVolume
    from .soft_volume import soft_volume, log_soft_volume, SoftVolume
    from .bessel_volume import (
        bessel_volume_approx,
        log_bessel_volume_approx,
        BesselApproxVolume,
    )

if tensorflow_is_available():
    from .tf_volume import (
        TFVolume,
        tf_hard_volume,
        tf_log_hard_volume,
        TFHardVolume,
    )
    from .tf_soft_volume import (
        tf_soft_volume,
        tf_log_soft_volume,
        TFSoftVolume,
    )

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
