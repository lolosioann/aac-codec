"""
Type definitions for AAC codec.

Centralizes all type hints and type aliases used across the codebase.
"""

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

# Frame types
FrameType = Literal["OLS", "LSS", "ESH", "LPS"]

# Window types
WindowType = Literal["KBD", "SIN"]

# Type aliases for clarity
StereoFrame = NDArray[np.float64]  # Shape: (2048, 2)
MonoFrame = NDArray[np.float64]  # Shape: (2048,)
MDCTCoeffsLong = NDArray[np.float64]  # Shape: (1024,) or (1024, 1)
MDCTCoeffsShort = NDArray[np.float64]  # Shape: (128, 8)


class ChannelData(TypedDict):
    """MDCT coefficients for a single channel."""

    frame_F: NDArray[np.float64]  # (1024, 1) for long, (128, 8) for short


class EncodedFrame(TypedDict):
    """
    Structure for a single encoded frame in Part 1.

    Attributes
    ----------
    frame_type : FrameType
        Type of frame: "OLS", "LSS", "ESH", "LPS"
    win_type : WindowType
        Window type used: "KBD" or "SIN"
    chl : ChannelData
        Left channel MDCT coefficients
    chr : ChannelData
        Right channel MDCT coefficients
    """

    frame_type: FrameType
    win_type: WindowType
    chl: ChannelData
    chr: ChannelData
