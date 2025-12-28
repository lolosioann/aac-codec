"""
Demo script for full AAC codec (Part 3).

Demonstrates complete encoding/decoding with compression metrics.
"""

import numpy as np
from numpy.typing import NDArray


def demo_aac_3(
    filename_in: str,
    filename_out: str,
    filename_aac_coded: str,
) -> tuple[float, float, float]:
    """
    Demonstrate full AAC codec and compute performance metrics.

    Encodes and decodes an audio file, measuring quality and compression.

    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    filename_out : str
        Path to output WAV file for decoded audio
    filename_aac_coded : str
        Path to save encoded sequence (.mat file)

    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB
    bitrate : float
        Average bitrate in bits per second
    compression : float
        Compression ratio (original_bitrate / coded_bitrate)

    Notes
    -----
    Processing steps:
    1. Load original audio
    2. Encode with aac_coder_3()
    3. Decode with i_aac_coder_3()
    4. Compute SNR between original and reconstructed
    5. Compute bitrate from encoded data
    6. Compute compression ratio
    7. Save reconstructed audio

    SNR Formula:
    SNR_dB = 10 * log10(sum(signal^2) / sum((signal - reconstructed)^2))

    Bitrate Calculation:
    bitrate = total_bits / duration_seconds
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


def _compute_bitrate(
    aac_seq: list,
    sample_rate: int,
    hop_size: int,
) -> float:
    """
    Compute average bitrate from encoded sequence.

    Parameters
    ----------
    aac_seq : list[EncodedFrame3]
        Encoded frame sequence
    sample_rate : int
        Sample rate in Hz
    hop_size : int
        Hop size in samples

    Returns
    -------
    bitrate : float
        Average bitrate in bits per second

    Notes
    -----
    - Counts bits in Huffman streams (sfc and stream fields)
    - Duration = num_frames * hop_size / sample_rate
    - Bitrate = total_bits / duration
    """
    raise NotImplementedError()


def _compute_compression_ratio(
    bitrate_original: float,
    bitrate_coded: float,
) -> float:
    """
    Compute compression ratio.

    Parameters
    ----------
    bitrate_original : float
        Original bitrate (e.g., 48000 Hz * 16 bits * 2 channels)
    bitrate_coded : float
        Coded bitrate from _compute_bitrate()

    Returns
    -------
    compression : float
        Compression ratio (original / coded)

    Notes
    -----
    Higher values indicate better compression.
    Typical values: 5-15x for perceptual audio coding
    """
    raise NotImplementedError()
