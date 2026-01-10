"""
Demo and Testing Module for AAC Level 1.

Demonstrates the encoding/decoding pipeline and computes quality metrics.
"""

import numpy as np

from .aac_coder_1 import _read_wav_file, aac_coder_1, i_aac_coder_1


def demo_aac_1(filename_in: str, filename_out: str) -> float:
    """
    Demonstrate AAC Level 1 encoding and decoding, compute SNR.

    Parameters
    ----------
    filename_in : str
        Input WAV file path (stereo, 48kHz)
    filename_out : str
        Output WAV file path for decoded audio

    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB

    Processing Steps
    ----------------
    1. Encode input file using aac_coder_1()
    2. Decode using i_aac_coder_1()
    3. Compute SNR between original and decoded
    4. Print results and statistics
    5. Return SNR value

    Notes
    -----
    Also prints useful information:
    - Number of frames encoded
    - Frame type distribution
    - Compression statistics (even though Part 1 doesn't compress)
    """
    print("=" * 60)
    print("AAC Level 1 Demo - Encoder/Decoder")
    print("=" * 60)

    # Step 1: Read original audio
    print(f"\nReading input file: {filename_in}")
    original, sample_rate = _read_wav_file(filename_in)
    print(f"  - Duration: {len(original) / sample_rate:.2f} seconds")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Channels: {original.shape[1]}")

    # Step 2: Encode
    print("\nEncoding...")
    aac_seq = aac_coder_1(filename_in)
    print(f"  - Encoded {len(aac_seq)} frames")

    # Step 3: Print encoding statistics
    _print_encoding_stats(aac_seq)

    # Step 4: Decode
    print("\nDecoding...")
    decoded = i_aac_coder_1(aac_seq, filename_out)
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


def _compute_snr(original: np.ndarray, decoded: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio between original and decoded signals.

    Parameters
    ----------
    original : np.ndarray
        Original audio samples, shape (num_samples, num_channels)
    decoded : np.ndarray
        Decoded audio samples, same shape as original

    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB

    Formula
    -------
    SNR = 10 * log10(sum(original^2) / sum((original - decoded)^2))

    Notes
    -----
    - Ensure both arrays have same length (trim/pad if needed)
    - Handle edge case where error is zero (infinite SNR)
    - Compute across all channels
    """
    signal_power = np.sum(original**2)
    noise = original - decoded
    noise_power = np.sum(noise**2)

    if noise_power < np.finfo(float).eps:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)


def _compute_snr_per_channel(original: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """
    Compute SNR separately for each channel.

    Parameters
    ----------
    original : np.ndarray
        Original audio, shape (num_samples, num_channels)
    decoded : np.ndarray
        Decoded audio, shape (num_samples, num_channels)

    Returns
    -------
    snr_per_channel : np.ndarray
        SNR in dB for each channel, shape (num_channels,)
    """
    num_channels = original.shape[1]
    snr_per_channel = np.zeros(num_channels)

    for ch in range(num_channels):
        snr_per_channel[ch] = _compute_snr(
            original[:, ch : ch + 1], decoded[:, ch : ch + 1]
        )

    return snr_per_channel


def _align_signals(
    original: np.ndarray, decoded: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align original and decoded signals to same length.

    Parameters
    ----------
    original : np.ndarray
        Original signal, shape (n_samples, n_channels)
    decoded : np.ndarray
        Decoded signal, shape (m_samples, n_channels)

    Returns
    -------
    original_aligned : np.ndarray
        Trimmed/padded original
    decoded_aligned : np.ndarray
        Trimmed/padded decoded

    Notes
    -----
    Takes minimum length and trims both to match
    """
    min_len = min(original.shape[0], decoded.shape[0])
    return original[:min_len], decoded[:min_len]


def _print_encoding_stats(aac_seq: list) -> None:
    """
    Print statistics about the encoding.

    Parameters
    ----------
    aac_seq : list
        Encoded sequence from aac_coder_1()

    Prints
    ------
    - Total number of frames
    - Frame type distribution (count and percentage)
    - Average MDCT coefficient magnitude
    - Min/max coefficient values
    """
    print("\nEncoding Statistics:")
    print("-" * 60)

    # Count frame types
    frame_types = _count_frame_types(aac_seq)
    total_frames = len(aac_seq)

    print(f"Total frames: {total_frames}")
    print("\nFrame type distribution:")
    for ftype, count in frame_types.items():
        percentage = (count / total_frames) * 100
        print(f"  {ftype}: {count} ({percentage:.1f}%)")

    # Coefficient statistics
    stats = _compute_coefficient_statistics(aac_seq)
    print("\nMDCT coefficient statistics:")
    print(f"  Mean magnitude: {stats['mean_magnitude']:.4f}")
    print(f"  Std deviation: {stats['std_deviation']:.4f}")
    print(f"  Max coefficient: {stats['max_coefficient']:.4f}")
    print(f"  Min coefficient: {stats['min_coefficient']:.4f}")


def _count_frame_types(aac_seq: list) -> dict:
    """
    Count occurrences of each frame type.

    Parameters
    ----------
    aac_seq : list
        Encoded sequence

    Returns
    -------
    counts : dict
        Dictionary mapping frame_type to count
        e.g., {"OLS": 45, "LSS": 2, "ESH": 5, "LPS": 3}
    """
    counts = {"OLS": 0, "LSS": 0, "ESH": 0, "LPS": 0}

    for frame in aac_seq:
        frame_type = frame["frame_type"]
        counts[frame_type] += 1

    return counts


def _compute_coefficient_statistics(aac_seq: list) -> dict:
    """
    Compute statistics on MDCT coefficients.

    Parameters
    ----------
    aac_seq : list
        Encoded sequence

    Returns
    -------
    stats : dict
        Statistics dictionary with keys:
        - "mean_magnitude": float
        - "max_coefficient": float
        - "min_coefficient": float
        - "std_deviation": float
    """
    all_coeffs = []

    for frame in aac_seq:
        # Collect coefficients from both channels
        all_coeffs.append(frame["chl"]["frame_F"].flatten())
        all_coeffs.append(frame["chr"]["frame_F"].flatten())

    all_coeffs = np.concatenate(all_coeffs)

    return {
        "mean_magnitude": np.mean(np.abs(all_coeffs)),
        "max_coefficient": np.max(all_coeffs),
        "min_coefficient": np.min(all_coeffs),
        "std_deviation": np.std(all_coeffs),
    }


if __name__ == "__main__":
    input_file = "input_stereo_48kHz.wav"
    output_file = "decoded_output.wav"
    demo_aac_1(input_file, output_file)
