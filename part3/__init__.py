"""
Part 3: Full AAC Codec with Psychoacoustic Model and Quantization.

This module implements the complete AAC encoder/decoder including:
- TNS (from Part 2)
- Psychoacoustic Model
- Quantization with perceptual masking
- Huffman coding
"""

from .aac_coder_3 import aac_coder_3, i_aac_coder_3
from .demo_aac_3 import demo_aac_3
from .filterbank import filter_bank, i_filter_bank
from .psychoacoustic import psycho
from .quantizer import aac_quantizer, i_aac_quantizer
from .ssc import SSC
from .tns import i_tns, tns
from .types import (
    ChannelData3,
    EncodedFrame3,
    FrameType,
    GlobalGain,
    QuantizedSymbols,
    ScaleFactors,
    SMRArray,
    ThresholdArray,
    WindowType,
)

__all__ = [
    "SSC",
    "filter_bank",
    "i_filter_bank",
    "tns",
    "i_tns",
    "psycho",
    "aac_quantizer",
    "i_aac_quantizer",
    "aac_coder_3",
    "i_aac_coder_3",
    "demo_aac_3",
    "FrameType",
    "WindowType",
    "ChannelData3",
    "EncodedFrame3",
    "SMRArray",
    "ThresholdArray",
    "QuantizedSymbols",
    "ScaleFactors",
    "GlobalGain",
]
