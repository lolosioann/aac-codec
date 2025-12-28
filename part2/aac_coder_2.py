"""
AAC Encoder/Decoder for Part 2 (with TNS).

Implements the Level 2 codec pipeline including TNS processing.
"""


import numpy as np
from numpy.typing import NDArray

from .types import EncodedFrame2


def aac_coder_2(filename_in: str) -> list[EncodedFrame2]:
    """
    Encode audio file using AAC Level 2 codec with TNS.

    Pipeline for each frame:
    1. SSC - Determine frame type
    2. Filterbank - MDCT transform
    3. TNS - Temporal noise shaping (for each channel)

    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)

    Returns
    -------
    aac_seq_2 : list[EncodedFrame2]
        Sequence of encoded frames, each containing:
        - frame_type: Frame type ("OLS", "LSS", "ESH", "LPS")
        - win_type: Window type ("KBD" or "SIN")
        - chl: Left channel data (frame_F, tns_coeffs)
        - chr: Right channel data (frame_F, tns_coeffs)

    Notes
    -----
    - Assumes input is stereo, 48kHz
    - Uses 50% overlapping frames (2048 samples, hop 1024)
    - TNS is applied independently to each channel
    """
    raise NotImplementedError()


def i_aac_coder_2(
    aac_seq_2: list[EncodedFrame2],
    filename_out: str,
) -> NDArray[np.float64]:
    """
    Decode AAC Level 2 sequence and save to WAV file.

    Pipeline for each frame (reverse):
    1. iTNS - Inverse temporal noise shaping (for each channel)
    2. iFilterbank - Inverse MDCT transform
    3. Overlap-add reconstruction

    Parameters
    ----------
    aac_seq_2 : list[EncodedFrame2]
        Sequence of encoded frames from aac_coder_2()
    filename_out : str
        Path to output WAV file (will be stereo, 48kHz)

    Returns
    -------
    x : NDArray[np.float64]
        Reconstructed audio signal, shape (num_samples, 2)

    Notes
    -----
    - Performs overlap-add to reconstruct continuous signal
    - Saves result to WAV file
    - Returns reconstructed signal for analysis
    """
    raise NotImplementedError()
