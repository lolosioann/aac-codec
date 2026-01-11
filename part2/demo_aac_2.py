"""
Demo script for AAC Level 2 codec (with TNS).

Demonstrates encoding/decoding and computes SNR.
"""

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .aac_coder_2 import aac_coder_2, i_aac_coder_2


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
    print("=" * 60)
    print("AAC Level 2 Demo - Encoder/Decoder with TNS")
    print("=" * 60)

    # Step 1: Read original audio
    print(f"\nReading input file: {filename_in}")
    original, sample_rate = sf.read(filename_in, dtype="float32")
    print(f"  - Duration: {len(original) / sample_rate:.2f} seconds")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Channels: {original.shape[1]}")

    # Step 2: Encode
    print("\nEncoding with TNS...")
    aac_seq = aac_coder_2(filename_in)
    print(f"  - Encoded {len(aac_seq)} frames")

    # Step 3: Print encoding statistics
    _print_encoding_stats(aac_seq)

    # Step 4: Decode
    print("\nDecoding...")
    decoded = i_aac_coder_2(aac_seq, filename_out)
    print(f"  - Decoded {len(decoded)} samples")
    print(f"  - Output file: {filename_out}")

    # Step 5: Align signals and compute SNR
    print("\nComputing quality metrics...")
    original_aligned, decoded_aligned = _align_signals(original, decoded)

    snr = _compute_snr(original_aligned, decoded_aligned)
    snr_per_ch = _compute_snr_per_channel(original_aligned, decoded_aligned)

    # Step 6: Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Overall SNR: {snr:.2f} dB")
    print(f"Left channel SNR: {snr_per_ch[0]:.2f} dB")
    print(f"Right channel SNR: {snr_per_ch[1]:.2f} dB")
    print("=" * 60)

    return snr


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
    signal_power = np.sum(original**2)
    noise = original - reconstructed
    noise_power = np.sum(noise**2)

    if noise_power < np.finfo(float).eps:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)


def _compute_snr_per_channel(
    original: NDArray[np.float64],
    reconstructed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute SNR separately for each channel."""
    num_channels = original.shape[1]
    snr_per_channel = np.zeros(num_channels)

    for ch in range(num_channels):
        snr_per_channel[ch] = _compute_snr(
            original[:, ch : ch + 1], reconstructed[:, ch : ch + 1]
        )

    return snr_per_channel


def _align_signals(
    original: NDArray[np.float64],
    decoded: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align original and decoded signals to same length."""
    min_len = min(original.shape[0], decoded.shape[0])
    return original[:min_len], decoded[:min_len]


def _print_encoding_stats(aac_seq: list) -> None:
    """Print statistics about the encoding."""
    print("\nEncoding Statistics:")
    print("-" * 60)

    # Count frame types
    frame_types = {"OLS": 0, "LSS": 0, "ESH": 0, "LPS": 0}
    total_frames = len(aac_seq)

    for frame in aac_seq:
        frame_type = frame["frame_type"]
        frame_types[frame_type] += 1

    print(f"Total frames: {total_frames}")
    print("\nFrame type distribution:")
    for ftype, count in frame_types.items():
        percentage = (count / total_frames) * 100
        print(f"  {ftype}: {count} ({percentage:.1f}%)")

    # TNS statistics
    tns_active_count = 0
    for frame in aac_seq:
        chl_coeffs = frame["chl"]["tns_coeffs"]
        chr_coeffs = frame["chr"]["tns_coeffs"]
        if not np.allclose(chl_coeffs, 0) or not np.allclose(chr_coeffs, 0):
            tns_active_count += 1

    print(
        f"\nTNS active in {tns_active_count}/{total_frames} frames "
        f"({100 * tns_active_count / total_frames:.1f}%)"
    )


if __name__ == "__main__":
    input_file = "part2/input_stereo_38kHz.wav"
    output_file = "part2/decoded_output.wav"
    demo_aac_2(input_file, output_file)
