"""
Type definitions for AAC codec Part 2.

Extends Part 1 types with TNS-specific structures.
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

# TNS coefficient array types
TNSCoeffsLong = NDArray[np.float64]  # Shape: (4, 1) for long frames
TNSCoeffsShort = NDArray[np.float64]  # Shape: (4, 8) for short frames


class ChannelData2(TypedDict):
    """MDCT coefficients and TNS data for a single channel (Part 2)."""

    frame_F: NDArray[np.float64]  # (1024, 1) for long, (128, 8) for short
    tns_coeffs: NDArray[np.float64]  # (4, 1) for long, (4, 8) for short


class EncodedFrame2(TypedDict):
    """
    Structure for a single encoded frame in Part 2.

    Attributes
    ----------
    frame_type : FrameType
        Type of frame: "OLS", "LSS", "ESH", "LPS"
    win_type : WindowType
        Window type used: "KBD" or "SIN"
    chl : ChannelData2
        Left channel MDCT coefficients and TNS coefficients
    chr : ChannelData2
        Right channel MDCT coefficients and TNS coefficients
    """

    frame_type: FrameType
    win_type: WindowType
    chl: ChannelData2
    chr: ChannelData2
