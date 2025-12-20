"""
Part 1 - Basic AAC Encoder/Decoder Implementation.

This module implements the foundational components of an AAC audio codec:
- Sequence Segmentation Control (SSC)
- Filterbank (MDCT/IMDCT)
- Basic encoder/decoder pipeline
"""

from . import constants
from .aac_coder_1 import aac_coder_1, i_aac_coder_1
from .demo_aac_1 import demo_aac_1
from .filterbank import filter_bank, i_filter_bank
from .ssc import SSC
from .types import ChannelData, EncodedFrame, FrameType, WindowType

__all__ = [
    # Main functions
    "SSC",
    "filter_bank",
    "i_filter_bank",
    "aac_coder_1",
    "i_aac_coder_1",
    "demo_aac_1",
    # Types
    "FrameType",
    "WindowType",
    "EncodedFrame",
    "ChannelData",
    # Constants module
    "constants",
]
