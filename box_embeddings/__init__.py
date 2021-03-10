import os
import importlib.util

box_debug_level = 0

if os.environ.get("BOX_DEBUG_LEVEL"):
    box_debug_level = int(os.environ["BOX_DEBUG_LEVEL"])

_torch_available = importlib.util.find_spec("torch") is not None
_tensorflow_available = importlib.util.find_spec("tensorflow") is not None


def torch_is_available():
    return _torch_available


def tensorflow_is_available():
    return _tensorflow_available
