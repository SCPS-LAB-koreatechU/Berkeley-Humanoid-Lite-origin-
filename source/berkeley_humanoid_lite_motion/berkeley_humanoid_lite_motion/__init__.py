"""Capture-to-robot motion tooling for Berkeley Humanoid Lite.

Turns retargeted human motion into something this robot can be trained on, and
is explicit about what the hardware drops on the way. Nothing here imports Isaac
Lab or ROS; it is numpy and scipy so it can run on a laptop, in CI, and inside a
simulation environment alike.
"""

from .augment import CaptureNoise, mirror, time_warp
from .dexhand import HandModel
from .hand_retarget import HandRetargeter, RetargetWeights
from .motion import MotionSequence
from .urdf import Chain

__all__ = [
    "CaptureNoise", "Chain", "HandModel", "HandRetargeter",
    "MotionSequence", "RetargetWeights", "mirror", "time_warp",
]
