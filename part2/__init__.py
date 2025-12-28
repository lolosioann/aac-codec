"""
Part 2: AAC Codec with TNS (Temporal Noise Shaping).

This module extends Part 1 with TNS processing for improved temporal noise shaping.
"""

from .aac_coder_2 import aac_coder_2, i_aac_coder_2
from .demo_aac_2 import demo_aac_2
from .filterbank import filter_bank, i_filter_bank
from .ssc import SSC
from .tns import i_tns, tns
from .types import ChannelData2, EncodedFrame2, FrameType, WindowType

__all__ = [
    "SSC",
    "filter_bank",
    "i_filter_bank",
    "tns",
    "i_tns",
    "aac_coder_2",
    "i_aac_coder_2",
    "demo_aac_2",
    "FrameType",
    "WindowType",
    "ChannelData2",
    "EncodedFrame2",
]
