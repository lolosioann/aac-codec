"""
Full AAC Encoder/Decoder (Part 3).

Implements complete AAC codec pipeline including psychoacoustic model,
quantization, and Huffman coding.
"""

from typing import Any

import numpy as np
import scipy.io as sio
import soundfile as sf
from numpy.typing import NDArray

from .aac_types import ChannelData3, EncodedFrame3, FrameType, WindowType
from .constants import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE, SCALEFACTOR_HUFFMAN_CODEBOOK
from .filterbank import filter_bank, i_filter_bank
from .huff_utils import decode_huff, encode_huff, load_LUT
from .psychoacoustic import B219a, B219b, psycho
from .quantizer import aac_quantizer, i_aac_quantizer
from .ssc import SSC
from .tns import i_tns, tns

# Load Huffman LUTs at module level
_huff_lut = load_LUT()


def aac_coder_3(filename_in: str, filename_aac_coded: str) -> list[EncodedFrame3]:
    """
    Encode audio file using full AAC codec.

    Pipeline for each frame:
    1. SSC - Determine frame type
    2. Filterbank - MDCT transform
    3. TNS - Temporal noise shaping (per channel)
    4. Psychoacoustic Model - Compute SMR (per channel)
    5. Quantizer - Quantize MDCT coefficients (per channel)
    6. Huffman - Encode scalefactors and quantized symbols (per channel)

    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    filename_aac_coded : str
        Path to save encoded sequence (.mat file)

    Returns
    -------
    aac_seq_3 : list[EncodedFrame3]
        Sequence of encoded frames, each containing:
        - frame_type: Frame type ("OLS", "LSS", "ESH", "LPS")
        - win_type: Window type ("KBD" or "SIN")
        - chl: Left channel data (tns_coeffs, T, G, sfc, stream, codebook)
        - chr: Right channel data (tns_coeffs, T, G, sfc, stream, codebook)

    Notes
    -----
    - Assumes input is stereo, 48kHz
    - Uses 50% overlapping frames (2048 samples, hop 1024)
    - Saves encoded sequence to .mat file
    - Returns sequence for further analysis
    """
    # Read WAV file
    samples, sample_rate = _read_wav_file(filename_in)

    # Split into overlapping frames
    frames = _split_into_frames(samples, FRAME_SIZE, HOP_SIZE)

    # Default window type
    win_type: WindowType = "KBD"

    # Initialize state
    aac_seq_3: list[EncodedFrame3] = []
    prev_frame_type: FrameType = "OLS"

    # Previous frames for psychoacoustic model (per channel)
    # Need 2 previous frames for temporal prediction
    prev_frames_L: list[NDArray[np.float64]] = [
        np.zeros(FRAME_SIZE),
        np.zeros(FRAME_SIZE),
    ]
    prev_frames_R: list[NDArray[np.float64]] = [
        np.zeros(FRAME_SIZE),
        np.zeros(FRAME_SIZE),
    ]

    for i, frame in enumerate(frames):
        # Get next frame for lookahead (or zeros if last frame)
        next_frame = frames[i + 1] if i + 1 < len(frames) else np.zeros_like(frame)

        # Encode this frame
        encoded = _encode_frame(
            frame,
            next_frame,
            prev_frame_type,
            win_type,
            prev_frames_L,
            prev_frames_R,
        )
        aac_seq_3.append(encoded)

        # Update previous frame type
        prev_frame_type = encoded["frame_type"]

        # Update previous frames for psychoacoustic (shift history)
        prev_frames_L = [frame[:, 0].copy(), prev_frames_L[0]]
        prev_frames_R = [frame[:, 1].copy(), prev_frames_R[0]]

    # Save to .mat file
    _save_encoded_sequence(aac_seq_3, filename_aac_coded)

    return aac_seq_3


def i_aac_coder_3(
    aac_seq_3: list[EncodedFrame3],
    filename_out: str,
) -> NDArray[np.float64]:
    """
    Decode full AAC sequence and save to WAV file.

    Pipeline for each frame (reverse):
    1. Huffman Decode - Recover scalefactors and quantized symbols (per channel)
    2. iQuantizer - Dequantize to MDCT coefficients (per channel)
    3. iTNS - Inverse temporal noise shaping (per channel)
    4. iFilterbank - Inverse MDCT transform
    5. Overlap-add reconstruction

    Parameters
    ----------
    aac_seq_3 : list[EncodedFrame3]
        Sequence of encoded frames from aac_coder_3()
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
    # Decode each frame
    decoded_frames = []
    for encoded_frame in aac_seq_3:
        frame_T = _decode_frame(encoded_frame)
        decoded_frames.append(frame_T)

    # Overlap-add to reconstruct signal
    samples = _overlap_add(decoded_frames, HOP_SIZE)

    # Write to WAV file
    _write_wav_file(filename_out, samples, SAMPLE_RATE)

    return samples


def _load_huffman_utils() -> tuple[Any, Any, Any]:
    """
    Load Huffman encoding/decoding utilities.

    Returns
    -------
    encode_huff : callable
        Huffman encoding function
    decode_huff : callable
        Huffman decoding function
    load_LUT : callable
        Function to load Huffman lookup tables

    Notes
    -----
    - Uses huff_utils.py (provided)
    - Codebook 11 for scalefactors
    - Automatic codebook selection for MDCT coefficients
    """
    return encode_huff, decode_huff, load_LUT


# =============================================================================
# Helper Functions
# =============================================================================


def _read_wav_file(filename: str) -> tuple[NDArray[np.float64], int]:
    """Read a WAV file and return samples and sample rate."""
    if not filename.endswith(".wav"):
        raise ValueError("Input file must be a WAV file.")

    with sf.SoundFile(filename) as f:
        sample_rate = f.samplerate
        samples = f.read(dtype="float64")

    return samples, sample_rate


def _write_wav_file(
    filename: str, samples: NDArray[np.float64], sample_rate: int
) -> None:
    """Write audio samples to a WAV file."""
    if not filename.endswith(".wav"):
        raise ValueError("Output file must be a WAV file.")

    sf.write(filename, samples, sample_rate)


def _split_into_frames(
    samples: NDArray[np.float64], frame_size: int, hop_size: int
) -> list[NDArray[np.float64]]:
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


def _overlap_add(
    frames: list[NDArray[np.float64]], hop_size: int
) -> NDArray[np.float64]:
    """Reconstruct signal from overlapping frames."""
    frame_size = frames[0].shape[0]
    num_samples = hop_size * (len(frames) - 1) + frame_size
    samples = np.zeros((num_samples, 2))

    for i, frame in enumerate(frames):
        start = i * hop_size
        end = start + frame_size
        samples[start:end] += frame

    return samples


def _save_encoded_sequence(aac_seq: list[EncodedFrame3], filename: str) -> None:
    """Save encoded sequence to .mat file."""
    # Convert to format suitable for scipy.io.savemat
    # Store as structured array
    data = {"aac_seq_3": aac_seq}
    sio.savemat(filename, data, do_compression=True)


def _compute_threshold_from_smr(
    X: NDArray[np.float64],
    SMR: NDArray[np.float64],
    band_table: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute perceptual threshold T(b) = P(b) / SMR(b)."""
    num_bands = band_table.shape[0]
    P = np.zeros(num_bands)

    for b in range(num_bands):
        w_low = int(band_table[b, 1])
        w_high = int(band_table[b, 2])
        P[b] = np.sum(X[w_low : w_high + 1] ** 2)

    # Avoid division by zero
    SMR_safe = np.maximum(SMR.flatten(), 1e-10)
    T = P / SMR_safe

    return T


def _encode_frame(
    frame: NDArray[np.float64],
    next_frame: NDArray[np.float64],
    prev_frame_type: FrameType,
    win_type: WindowType,
    prev_frames_L: list[NDArray[np.float64]],
    prev_frames_R: list[NDArray[np.float64]],
) -> EncodedFrame3:
    """
    Encode a single frame with full pipeline.

    Parameters
    ----------
    frame : NDArray[np.float64]
        Current frame, shape (2048, 2)
    next_frame : NDArray[np.float64]
        Next frame for lookahead, shape (2048, 2)
    prev_frame_type : FrameType
        Previous frame type
    win_type : WindowType
        Window type to use
    prev_frames_L : list[NDArray[np.float64]]
        Previous 2 frames for left channel [prev_1, prev_2]
    prev_frames_R : list[NDArray[np.float64]]
        Previous 2 frames for right channel [prev_1, prev_2]

    Returns
    -------
    encoded_frame : EncodedFrame3
        Fully encoded frame
    """
    # Step 1: Determine frame type using SSC
    frame_type = SSC(frame, next_frame, prev_frame_type)

    # Step 2: Apply filterbank to get MDCT coefficients
    frame_F = filter_bank(frame, frame_type, win_type)

    # Step 3-6: Process each channel
    if frame_type == "ESH":
        # ESH: frame_F shape is (128, 8, 2)
        chl_F = frame_F[:, :, 0]  # (128, 8)
        chr_F = frame_F[:, :, 1]  # (128, 8)
    else:
        # Long frames: frame_F shape is (1024, 2)
        chl_F = frame_F[:, 0:1]  # (1024, 1)
        chr_F = frame_F[:, 1:2]  # (1024, 1)

    # Encode left channel
    chl_data = _encode_channel(
        chl_F,
        frame[:, 0],
        frame_type,
        prev_frames_L[0],
        prev_frames_L[1],
    )

    # Encode right channel
    chr_data = _encode_channel(
        chr_F,
        frame[:, 1],
        frame_type,
        prev_frames_R[0],
        prev_frames_R[1],
    )

    encoded_frame: EncodedFrame3 = {
        "frame_type": frame_type,
        "win_type": win_type,
        "chl": chl_data,
        "chr": chr_data,
    }

    return encoded_frame


def _encode_channel(
    frame_F: NDArray[np.float64],
    frame_T: NDArray[np.float64],
    frame_type: FrameType,
    prev_frame_1: NDArray[np.float64],
    prev_frame_2: NDArray[np.float64],
) -> ChannelData3:
    """
    Encode a single channel through TNS, psychoacoustic, quantizer, and Huffman.

    Parameters
    ----------
    frame_F : NDArray[np.float64]
        MDCT coefficients, (1024, 1) or (128, 8)
    frame_T : NDArray[np.float64]
        Time-domain samples, (2048,)
    frame_type : FrameType
        Frame type
    prev_frame_1 : NDArray[np.float64]
        Previous frame (i-1), (2048,)
    prev_frame_2 : NDArray[np.float64]
        Pre-previous frame (i-2), (2048,)

    Returns
    -------
    channel_data : ChannelData3
        Encoded channel data
    """
    # Step 3: Apply TNS
    frame_F_tns, tns_coeffs = tns(frame_F, frame_type)

    # Step 4: Compute SMR using psychoacoustic model
    SMR = psycho(frame_T, frame_type, prev_frame_1, prev_frame_2)

    # Step 5: Quantize
    S, sfc, G = aac_quantizer(frame_F_tns, frame_type, SMR)

    # Compute perceptual threshold T for visualization
    if frame_type == "ESH":
        band_table = B219b
        # For ESH, compute T for each subframe and average (or use last)
        T = _compute_threshold_from_smr(
            frame_F_tns[:, -1].flatten(), SMR[:, -1], band_table
        )
    else:
        band_table = B219a
        T = _compute_threshold_from_smr(
            frame_F_tns.flatten(), SMR.flatten(), band_table
        )

    # Step 6: Huffman encode
    # Encode scalefactor deltas with codebook 11
    # sfc[0] = G (global gain) is stored separately, so only encode sfc[1:]
    sfc_deltas = np.round(sfc[1:]).astype(np.int32)
    # force_codebook returns only the bitstream, not a tuple
    sfc_stream = encode_huff(
        sfc_deltas, _huff_lut, force_codebook=SCALEFACTOR_HUFFMAN_CODEBOOK
    )

    # Encode quantized symbols with auto-selected codebook
    stream, codebook = encode_huff(S, _huff_lut)

    channel_data: ChannelData3 = {
        "tns_coeffs": tns_coeffs,
        "T": T,
        "G": G,
        "sfc": sfc_stream,
        "stream": stream,
        "codebook": codebook,
    }

    return channel_data


def _decode_frame(encoded_frame: EncodedFrame3) -> NDArray[np.float64]:
    """
    Decode a single frame with full inverse pipeline.

    Parameters
    ----------
    encoded_frame : EncodedFrame3
        Encoded frame from aac_coder_3()

    Returns
    -------
    frame_T : NDArray[np.float64]
        Decoded time-domain frame, shape (2048, 2)
    """
    frame_type = encoded_frame["frame_type"]
    win_type = encoded_frame["win_type"]

    # Decode each channel
    chl_F = _decode_channel(encoded_frame["chl"], frame_type)
    chr_F = _decode_channel(encoded_frame["chr"], frame_type)

    # Reconstruct frame_F for i_filter_bank
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

    # Apply inverse filterbank
    frame_T = i_filter_bank(frame_F, frame_type, win_type)

    return frame_T


def _decode_channel(
    channel_data: ChannelData3,
    frame_type: FrameType,
) -> NDArray[np.float64]:
    """
    Decode a single channel through Huffman, dequantizer, and inverse TNS.

    Parameters
    ----------
    channel_data : ChannelData3
        Encoded channel data
    frame_type : FrameType
        Frame type

    Returns
    -------
    frame_F : NDArray[np.float64]
        Reconstructed MDCT coefficients, (1024,) or (128, 8)
    """
    # Step 1: Huffman decode scalefactors
    # We encoded sfc[1:] (deltas), so decode and prepend 0 for sfc[0]
    # (sfc[0] is overridden by G in _dpcm_to_scalefactor anyway)
    sfc_LUT = _huff_lut[SCALEFACTOR_HUFFMAN_CODEBOOK]
    sfc_decoded = decode_huff(channel_data["sfc"], sfc_LUT)

    # Determine expected sfc length (deltas = num_bands - 1)
    num_bands = 42 if frame_type == "ESH" else 69
    sfc_deltas = np.array(sfc_decoded[: num_bands - 1], dtype=np.float64)
    sfc = np.concatenate([[0.0], sfc_deltas])  # Prepend 0 for sfc[0]

    # Step 2: Huffman decode quantized symbols
    codebook = channel_data["codebook"]
    if codebook == 0:
        # All zeros
        S = np.zeros(1024, dtype=np.int32)
    else:
        stream_LUT = _huff_lut[codebook]
        S_decoded = decode_huff(channel_data["stream"], stream_LUT)
        S = np.array(S_decoded[:1024], dtype=np.int32)

    # Step 3: Dequantize
    G = channel_data["G"]
    frame_F_tns = i_aac_quantizer(S, sfc, G, frame_type)

    # Step 4: Inverse TNS
    tns_coeffs = channel_data["tns_coeffs"]
    frame_F = i_tns(frame_F_tns, frame_type, tns_coeffs)

    return frame_F
