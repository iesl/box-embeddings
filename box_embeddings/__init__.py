import os

box_debug_level = 0

if os.environ.get("BOX_DEBUG_LEVEL"):
    box_debug_level = int(os.environ["BOX_DEBUG_LEVEL"])
