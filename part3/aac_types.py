"""
Type definitions for AAC codec Part 3.

Extends Part 2 types with psychoacoustic and quantization structures.
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

# Psychoacoustic types
SMRArray = NDArray[np.float64]  # Shape: (69, 1) or (42, 8) - Signal to Mask Ratio
ThresholdArray = NDArray[np.float64]  # Shape: (69,) or (42,) - Perceptual thresholds

# Quantization types
QuantizedSymbols = NDArray[np.int32]  # Shape: (1024,) - Quantized MDCT coefficients
ScaleFactors = NDArray[np.float64]  # Shape: (69,) or (42,) - Scale factors per band
GlobalGain = float | NDArray[np.float64]  # scalar for long, (8,) for short frames


class ChannelData3(TypedDict):
    """MDCT coefficients and all encoding data for a single channel (Part 3)."""

    tns_coeffs: NDArray[np.float64]  # (4, 1) for long, (4, 8) for short
    T: NDArray[np.float64]  # Perceptual thresholds (for visualization)
    G: float | NDArray[np.float64]  # Global gain (scalar or array of 8)
    sfc: str  # Huffman-encoded scale factors (binary string)
    stream: str  # Huffman-encoded quantized MDCT coefficients (binary string)
    codebook: int  # Huffman codebook index used


class EncodedFrame3(TypedDict):
    """
    Structure for a single encoded frame in Part 3 (full codec).

    Attributes
    ----------
    frame_type : FrameType
        Type of frame: "OLS", "LSS", "ESH", "LPS"
    win_type : WindowType
        Window type used: "KBD" or "SIN"
    chl : ChannelData3
        Left channel full encoding data
    chr : ChannelData3
        Right channel full encoding data
    """

    frame_type: FrameType
    win_type: WindowType
    chl: ChannelData3
    chr: ChannelData3
