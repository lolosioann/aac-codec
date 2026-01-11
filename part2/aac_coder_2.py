"""
AAC Encoder/Decoder for Part 2 (with TNS).

Implements the Level 2 codec pipeline including TNS processing.
"""

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .aac_types import EncodedFrame2, FrameType, WindowType
from .filterbank import filter_bank, i_filter_bank
from .ssc import SSC
from .tns import i_tns, tns


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
    from .constants import FRAME_SIZE, HOP_SIZE

    # Step 1: Read WAV file
    samples, sample_rate = _read_wav_file(filename_in)

    # Step 2: Split into overlapping frames
    frames = _split_into_frames(samples, FRAME_SIZE, HOP_SIZE)

    # Step 3: Get default window type
    win_type: WindowType = "KBD"

    # Step 4: Encode each frame
    aac_seq_2: list[EncodedFrame2] = []
    prev_frame_type: FrameType = "OLS"

    for i, frame in enumerate(frames):
        # Get next frame for lookahead (or use zeros if last frame)
        next_frame = frames[i + 1] if i + 1 < len(frames) else np.zeros_like(frame)

        # Encode this frame
        encoded = _encode_frame(frame, next_frame, prev_frame_type, win_type)
        aac_seq_2.append(encoded)

        # Update previous frame type
        prev_frame_type = encoded["frame_type"]

    return aac_seq_2


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
    from .constants import HOP_SIZE, SAMPLE_RATE

    # Step 1: Decode each frame
    decoded_frames = []
    for encoded_frame in aac_seq_2:
        frame_T = _decode_frame(encoded_frame)
        decoded_frames.append(frame_T)

    # Step 2: Overlap-add to reconstruct signal
    samples = _overlap_add(decoded_frames, HOP_SIZE)

    # Step 3: Write to WAV file
    _write_wav_file(filename_out, samples, SAMPLE_RATE)

    return samples


def _read_wav_file(filename: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return samples and sample rate."""
    if not filename.endswith(".wav"):
        raise ValueError("Input file must be a WAV file.")

    with sf.SoundFile(filename) as f:
        sample_rate = f.samplerate
        samples = f.read(dtype="float32")

    return samples, sample_rate


def _write_wav_file(filename: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write audio samples to a WAV file."""
    if not filename.endswith(".wav"):
        raise ValueError("Output file must be a WAV file.")

    sf.write(filename, samples, sample_rate)


def _split_into_frames(
    samples: np.ndarray, frame_size: int, hop_size: int
) -> list[np.ndarray]:
    """Split audio into overlapping frames."""
    frames = []
    num_samples = samples.shape[0]

    for start in range(0, num_samples, hop_size):
        end = min(start + frame_size, num_samples)
        frame = samples[start:end]

        if frame.shape[0] < frame_size:
            pad_width = frame_size - frame.shape[0]
            frame = np.pad(frame, ((0, pad_width), (0, 0)), mode="constant")

        frames.append(frame)

    return frames


def _overlap_add(frames: list[np.ndarray], hop_size: int) -> np.ndarray:
    """Reconstruct signal from overlapping frames."""
    frame_size = frames[0].shape[0]
    num_samples = hop_size * (len(frames) - 1) + frame_size
    samples = np.zeros((num_samples, 2))

    for i, frame in enumerate(frames):
        start = i * hop_size
        end = start + frame_size
        samples[start:end] += frame

    return samples


def _encode_frame(
    frame: np.ndarray,
    next_frame: np.ndarray,
    prev_frame_type: FrameType,
    win_type: WindowType,
) -> EncodedFrame2:
    """
    Encode a single frame with TNS.

    Parameters
    ----------
    frame : np.ndarray
        Current frame, shape (2048, 2)
    next_frame : np.ndarray
        Next frame for lookahead, shape (2048, 2)
    prev_frame_type : FrameType
        Previous frame type
    win_type : WindowType
        Window type to use

    Returns
    -------
    encoded_frame : EncodedFrame2
        Encoded frame with TNS coefficients
    """
    # Step 1: Determine frame type using SSC
    frame_type = SSC(frame, next_frame, prev_frame_type)

    # Step 2: Apply filterbank to get MDCT coefficients
    frame_F = filter_bank(frame, frame_type, win_type)

    # Step 3: Apply TNS to each channel
    if frame_type == "ESH":
        # ESH: frame_F shape is (128, 8, 2)
        chl_F = frame_F[:, :, 0]  # (128, 8)
        chr_F = frame_F[:, :, 1]  # (128, 8)
    else:
        # Long frames: frame_F shape is (1024, 2)
        chl_F = frame_F[:, 0:1]  # (1024, 1)
        chr_F = frame_F[:, 1:2]  # (1024, 1)

    # Apply TNS
    chl_F_tns, chl_tns_coeffs = tns(chl_F, frame_type)
    chr_F_tns, chr_tns_coeffs = tns(chr_F, frame_type)

    # Step 4: Build encoded frame
    encoded_frame: EncodedFrame2 = {
        "frame_type": frame_type,
        "win_type": win_type,
        "chl": {
            "frame_F": chl_F_tns,
            "tns_coeffs": chl_tns_coeffs,
        },
        "chr": {
            "frame_F": chr_F_tns,
            "tns_coeffs": chr_tns_coeffs,
        },
    }

    return encoded_frame


def _decode_frame(encoded_frame: EncodedFrame2) -> np.ndarray:
    """
    Decode a single frame with inverse TNS.

    Parameters
    ----------
    encoded_frame : EncodedFrame2
        Encoded frame with TNS coefficients

    Returns
    -------
    frame_T : np.ndarray
        Decoded time-domain frame, shape (2048, 2)
    """
    frame_type = encoded_frame["frame_type"]
    win_type = encoded_frame["win_type"]

    # Step 1: Apply inverse TNS
    chl_F_tns = encoded_frame["chl"]["frame_F"]
    chr_F_tns = encoded_frame["chr"]["frame_F"]
    chl_tns_coeffs = encoded_frame["chl"]["tns_coeffs"]
    chr_tns_coeffs = encoded_frame["chr"]["tns_coeffs"]

    chl_F = i_tns(chl_F_tns, frame_type, chl_tns_coeffs)
    chr_F = i_tns(chr_F_tns, frame_type, chr_tns_coeffs)

    # Step 2: Reconstruct frame_F for i_filter_bank
    if frame_type == "ESH":
        # ESH: shape (128, 8, 2)
        frame_F = np.zeros((128, 8, 2))
        frame_F[:, :, 0] = chl_F
        frame_F[:, :, 1] = chr_F
    else:
        # Long frames: shape (1024, 2)
        frame_F = np.zeros((1024, 2))
        frame_F[:, 0] = chl_F.flatten()
        frame_F[:, 1] = chr_F.flatten()

    # Step 3: Apply inverse filterbank
    frame_T = i_filter_bank(frame_F, frame_type, win_type)

    return frame_T
