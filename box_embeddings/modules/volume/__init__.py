import logging
from ... import torch_is_available, tensorflow_is_available

logger = logging.getLogger(__name__)

if torch_is_available():
    from .volume import Volume
    from .hard_volume import hard_volume, HardVolume
    from .soft_volume import soft_volume, SoftVolume
    from .bessel_volume import (
        bessel_volume_approx,
        BesselApproxVolume,
    )

if tensorflow_is_available():
    from .tf_volume import TFVolume
    from .tf_hard_volume import tf_hard_volume, TFHardVolume
    from .tf_soft_volume import tf_soft_volume, TFSoftVolume
    from .tf_bessel_volume import (
        tf_bessel_volume_approx,
        TFBesselApproxVolume,
    )

if not torch_is_available() and not tensorflow_is_available():
    logger.warning("Can't find versions of Pytorch or Tensorflow")
