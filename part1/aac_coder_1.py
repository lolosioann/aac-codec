"""
AAC Coder Level 1 - Main Encoder and Decoder.

Implements the complete encoding and decoding pipeline for Part 1.
"""

from typing import Any

import numpy as np
import soundfile as sf

from .filterbank import WindowType, filter_bank, i_filter_bank
from .ssc import SSC, FrameType


def aac_coder_1(filename_in: str) -> list[dict[str, Any]]:
    """
    Encode a WAV file using AAC Level 1 (SSC + Filterbank only).

    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)

    Returns
    -------
    aac_seq_1 : List[Dict[str, Any]]
        List of K encoded frames, where K is the number of frames.
        Each element is a dictionary with:
        {
            "frame_type": FrameType,  # "OLS", "LSS", "ESH", "LPS"
            "win_type": WindowType,    # "KBD" or "SIN"
            "chl": {
                "frame_F": np.ndarray  # MDCT coefficients for left channel
                                       # shape (1024, 1) for long frames
                                       # shape (128, 8) for ESH frames
            },
            "chr": {
                "frame_F": np.ndarray  # MDCT coefficients for right channel
            }
        }

    Notes
    -----
    Processing steps:
    1. Read WAV file (stereo, 48kHz)
    2. Split into overlapping frames (2048 samples, 50% overlap = 1024 hop)
    3. For each frame:
       a. Determine frame_type using SSC
       b. Apply filter_bank to get MDCT coefficients
       c. Store in dictionary structure
    """
    from .constants import FRAME_SIZE, HOP_SIZE

    # Step 1: Read WAV file
    samples, sample_rate = _read_wav_file(filename_in)

    # Step 2: Split into overlapping frames
    frames = _split_into_frames(samples, FRAME_SIZE, HOP_SIZE)

    # Step 3: Get default window type
    win_type = _get_default_window_type()

    # Step 4: Encode each frame
    aac_seq_1 = []
    prev_frame_type: FrameType = "OLS"  # Start with OLS

    for i, frame in enumerate(frames):
        # Get next frame for lookahead (or use zeros if last frame)
        next_frame = frames[i + 1] if i + 1 < len(frames) else np.zeros_like(frame)

        # Encode this frame
        encoded = _encode_frame(frame, next_frame, prev_frame_type, win_type)
        aac_seq_1.append(encoded)

        # Update previous frame type for next iteration
        prev_frame_type = encoded["frame_type"]

    return aac_seq_1


def i_aac_coder_1(aac_seq_1: list[dict[str, Any]], filename_out: str) -> np.ndarray:
    """
    Decode AAC Level 1 sequence back to audio.

    Parameters
    ----------
    aac_seq_1 : List[Dict[str, Any]]
        Encoded sequence from aac_coder_1()
    filename_out : str
        Path to output WAV file (will be stereo, 48kHz)

    Returns
    -------
    x : np.ndarray
        Decoded audio samples, shape (num_samples, 2)

    Notes
    -----
    Processing steps:
    1. For each frame in aac_seq_1:
       a. Apply i_filter_bank to get time-domain samples
       b. Overlap-add with 50% overlap
    2. Write result to WAV file
    3. Return samples array
    """
    from .constants import HOP_SIZE, SAMPLE_RATE

    # Step 1: Decode each frame to time domain
    decoded_frames = []
    for encoded_frame in aac_seq_1:
        frame_T = _decode_frame(encoded_frame)
        decoded_frames.append(frame_T)

    # Step 2: Overlap-add to reconstruct full signal
    samples = _overlap_add(decoded_frames, HOP_SIZE)

    # Step 3: Write to WAV file
    _write_wav_file(filename_out, samples, SAMPLE_RATE)

    return samples


def _read_wav_file(filename: str) -> tuple[np.ndarray, int]:
    """
    Read a WAV file and return samples and sample rate.

    Parameters
    ----------
    filename : str
        Path to WAV file

    Returns
    -------
    samples : np.ndarray
        Audio samples, shape (num_samples, num_channels)
        Normalized to float in range [-1, 1]
    sample_rate : int
        Sample rate in Hz

    Notes
    -----
    Use soundfile library: sf.read(filename)
    """
    if not filename.endswith(".wav"):
        raise ValueError("Input file must be a WAV file with .wav extension.")

    with sf.SoundFile(filename) as f:
        sample_rate = f.samplerate
        samples = f.read(dtype="float32")

    return samples, sample_rate


def _write_wav_file(filename: str, samples: np.ndarray, sample_rate: int) -> None:
    """
    Write audio samples to a WAV file.

    Parameters
    ----------
    filename : str
        Output file path
    samples : np.ndarray
        Audio samples, shape (num_samples, num_channels)
    sample_rate : int
        Sample rate in Hz

    Notes
    -----
    Use soundfile library: sf.write(filename, samples, sample_rate)
    """
    if not filename.endswith(".wav"):
        raise ValueError("Output file must be a WAV file with .wav extension.")

    sf.write(filename, samples, sample_rate)


def _split_into_frames(
    samples: np.ndarray, frame_size: int, hop_size: int
) -> list[np.ndarray]:
    """
    Split audio into overlapping frames.

    Parameters
    ----------
    samples : np.ndarray
        Input audio, shape (num_samples, 2)
    frame_size : int
        Frame length in samples (2048)
    hop_size : int
        Hop size in samples (1024 for 50% overlap)

    Returns
    -------
    frames : List[np.ndarray]
        List of frames, each shape (frame_size, 2)

    Notes
    -----
    May need to zero-pad the last frame if not enough samples remain
    """
    frames = []
    num_samples = samples.shape[0]
    for start in range(0, num_samples, hop_size):
        end = start + frame_size if start + frame_size <= num_samples else num_samples
        frame = samples[start:end]
        if frame.shape[0] < frame_size:
            # Zero-pad last frame if needed
            pad_width = frame_size - frame.shape[0]
            frame = np.pad(frame, ((0, pad_width), (0, 0)), mode="constant")
        frames.append(frame)
    return frames


def _overlap_add(frames: list[np.ndarray], hop_size: int) -> np.ndarray:
    """
    Reconstruct signal from overlapping frames using overlap-add.

    Parameters
    ----------
    frames : List[np.ndarray]
        List of time-domain frames, each shape (frame_size, 2)
    hop_size : int
        Hop size in samples (1024)

    Returns
    -------
    samples : np.ndarray
        Reconstructed audio, shape (num_samples, 2)

    Notes
    -----
    Overlapping regions are summed together. This works correctly
    with MDCT windows that satisfy the Princen-Bradley condition.
    """
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
) -> dict[str, Any]:
    """
    Encode a single frame.

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
    encoded_frame : Dict[str, Any]
        Dictionary with frame_type, win_type, and MDCT coefficients
    """
    # Step 1: Determine frame type using SSC
    frame_type = SSC(frame, next_frame, prev_frame_type)

    # Step 2: Apply filterbank to get MDCT coefficients
    frame_F = filter_bank(frame, frame_type, win_type)

    # Step 3: Build encoded frame dictionary
    encoded_frame = {
        "frame_type": frame_type,
        "win_type": win_type,
        "chl": {"frame_F": frame_F[:, :, 0] if frame_type == "ESH" else frame_F[:, 0]},
        "chr": {"frame_F": frame_F[:, :, 1] if frame_type == "ESH" else frame_F[:, 1]},
    }

    return encoded_frame


def _decode_frame(encoded_frame: dict[str, Any]) -> np.ndarray:
    """
    Decode a single frame.

    Parameters
    ----------
    encoded_frame : Dict[str, Any]
        Encoded frame from aac_seq_1

    Returns
    -------
    frame : np.ndarray
        Decoded time-domain frame, shape (2048, 2)
    """
    # Extract frame info
    frame_type = encoded_frame["frame_type"]
    win_type = encoded_frame["win_type"]

    # Reconstruct frame_F in the format expected by i_filter_bank
    if frame_type == "ESH":
        # For ESH: shape (128, 8, 2)
        frame_F = np.zeros((128, 8, 2))
        frame_F[:, :, 0] = encoded_frame["chl"]["frame_F"]
        frame_F[:, :, 1] = encoded_frame["chr"]["frame_F"]
    else:
        # For OLS, LSS, LPS: shape (1024, 2)
        frame_F = np.zeros((1024, 2))
        frame_F[:, 0] = encoded_frame["chl"]["frame_F"]
        frame_F[:, 1] = encoded_frame["chr"]["frame_F"]

    # Apply inverse filterbank
    frame_T = i_filter_bank(frame_F, frame_type, win_type)

    return frame_T


def _get_default_window_type() -> WindowType:
    """
    Get the default window type for Part 1.

    Returns
    -------
    win_type : WindowType
        Default is "KBD"

    Notes
    -----
    Part 1 assumes consistent window type throughout encoding.
    Can be modified to support dynamic switching in later parts.
    """
    return "KBD"
