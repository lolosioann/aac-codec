"""
Demo script for full AAC codec (Part 3).

Demonstrates complete encoding/decoding with compression metrics.
"""

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .aac_coder_3 import aac_coder_3, i_aac_coder_3
from .aac_types import EncodedFrame3
from .constants import HOP_SIZE


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
    # Step 1: Load original audio
    original, sample_rate = sf.read(filename_in, dtype="float64")

    # Step 2: Encode
    aac_seq = aac_coder_3(filename_in, filename_aac_coded)

    # Step 3: Decode
    reconstructed = i_aac_coder_3(aac_seq, filename_out)

    # Step 4: Compute SNR (align lengths)
    min_len = min(len(original), len(reconstructed))
    snr = _compute_snr(original[:min_len], reconstructed[:min_len])

    # Step 5: Compute bitrate
    bitrate = _compute_bitrate(aac_seq, sample_rate, HOP_SIZE)

    # Step 6: Compute compression ratio
    # Original: 16-bit stereo PCM
    original_bitrate = sample_rate * 16 * 2
    compression = _compute_compression_ratio(original_bitrate, bitrate)

    return snr, bitrate, compression


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
    noise = original - reconstructed
    signal_power = np.sum(original**2)
    noise_power = np.sum(noise**2)

    if noise_power < 1e-10:
        return float("inf")

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_db)


def _compute_bitrate(
    aac_seq: list[EncodedFrame3],
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
    total_bits = 0

    for frame in aac_seq:
        # Left channel
        total_bits += len(frame["chl"]["sfc"])
        total_bits += len(frame["chl"]["stream"])

        # Right channel
        total_bits += len(frame["chr"]["sfc"])
        total_bits += len(frame["chr"]["stream"])

    # Duration in seconds
    num_frames = len(aac_seq)
    duration = num_frames * hop_size / sample_rate

    if duration < 1e-10:
        return 0.0

    bitrate = total_bits / duration
    return bitrate


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
    if bitrate_coded < 1e-10:
        return float("inf")

    return bitrate_original / bitrate_coded


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m part3.demo_aac_3 <input.wav> [output.wav] [encoded.mat]"
        )
        print("Example: python -m part3.demo_aac_3 audio.wav decoded.wav encoded.mat")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "decoded_output.wav"
    encoded_file = sys.argv[3] if len(sys.argv) > 3 else "encoded.mat"

    print("=" * 60)
    print("AAC Level 3 Demo - Full Codec")
    print("=" * 60)
    print(f"\nInput:   {input_file}")
    print(f"Output:  {output_file}")
    print(f"Encoded: {encoded_file}")
    print("\nProcessing...")

    snr, bitrate, compression = demo_aac_3(input_file, output_file, encoded_file)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"SNR:               {snr:.2f} dB")
    print(f"Bitrate:           {bitrate / 1000:.2f} kbps")
    print(f"Compression ratio: {compression:.2f}x")
    print("=" * 60)
