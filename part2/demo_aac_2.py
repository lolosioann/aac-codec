"""
Demo script for AAC Level 2 codec (with TNS).

Demonstrates encoding/decoding and computes SNR.
"""

import numpy as np
from numpy.typing import NDArray


def demo_aac_2(filename_in: str, filename_out: str) -> float:
    """
    Demonstrate AAC Level 2 codec and compute SNR.

    Encodes and decodes an audio file, measuring reconstruction quality.

    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    filename_out : str
        Path to output WAV file for decoded audio

    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB

    Notes
    -----
    Processing steps:
    1. Load original audio
    2. Encode with aac_coder_2()
    3. Decode with i_aac_coder_2()
    4. Compute SNR between original and reconstructed
    5. Save reconstructed audio

    SNR Formula:
    SNR_dB = 10 * log10(sum(signal^2) / sum((signal - reconstructed)^2))
    """
    raise NotImplementedError()


def _compute_snr(
    original: NDArray[np.float64],
    reconstructed: NDArray[np.float64],
) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    original : NDArray[np.float64]
        Original signal, shape (num_samples, num_channels)
    reconstructed : NDArray[np.float64]
        Reconstructed signal, same shape as original

    Returns
    -------
    snr_db : float
        SNR in decibels

    Notes
    -----
    SNR = 10 * log10(signal_power / noise_power)
    where noise = original - reconstructed
    """
    raise NotImplementedError()
