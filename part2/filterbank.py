"""
Filterbank Module - MDCT/IMDCT Implementation.

Implements Modified Discrete Cosine Transform and windowing operations
for AAC audio encoding/decoding.
"""

from typing import Literal

import numpy as np

from .ssc import FrameType

WindowType = Literal["KBD", "SIN"]


def filter_bank(
    frame_T: np.ndarray, frame_type: FrameType, win_type: WindowType
) -> np.ndarray:
    """
    Apply windowing and MDCT transform to convert time-domain frame to frequency domain.

    Parameters
    ----------
    frame_T : np.ndarray
        Frame in time domain, shape (2048, 2) for stereo
    frame_type : FrameType
        Frame type: "OLS", "LSS", "ESH", or "LPS"
    win_type : WindowType
        Window type: "KBD" or "SIN"

    Returns
    -------
    frame_F : np.ndarray
        MDCT coefficients in frequency domain
        - For "OLS", "LSS", "LPS": shape (1024, 2)
        - For "ESH": shape (128, 8, 2) - 8 subframes of 128 coefficients each,
          organized as columns per subframe

    Notes
    -----
    Processing steps:
    1. Apply appropriate window based on frame_type
    2. Compute MDCT (produces N/2 coefficients from N samples)
    3. For ESH: process 8 overlapping subframes of 256 samples → 128 coefficients each
    """
    # Step 1: Create appropriate window for this frame type
    if frame_type == "ESH":
        # For EIGHT_SHORT, use short window (256 samples)
        from .constants import SHORT_FRAME_SIZE

        window = _create_window(SHORT_FRAME_SIZE, win_type, frame_type)
    else:
        # For OLS, LSS, LPS, use long window (2048 samples)
        from .constants import FRAME_SIZE

        window = _create_window(FRAME_SIZE, win_type, frame_type)

    # Step 2: Process each channel separately
    if frame_type == "ESH":
        # Process as 8 short subframes
        frame_F = np.zeros((128, 8, 2))
        for ch in range(2):
            frame_F[:, :, ch] = _process_short_frame(frame_T, window, ch)
    else:
        # Process as single long frame
        frame_F = np.zeros((1024, 2))
        for ch in range(2):
            frame_F[:, ch] = _process_long_frame(frame_T, window, ch)

    return frame_F


def i_filter_bank(
    frame_F: np.ndarray, frame_type: FrameType, win_type: WindowType
) -> np.ndarray:
    """
    Apply inverse MDCT and windowing to convert frequency-domain coefficients to time domain.

    Parameters
    ----------
    frame_F : np.ndarray
        MDCT coefficients
        - For "OLS", "LSS", "LPS": shape (1024, 2)
        - For "ESH": shape (128, 8, 2)
    frame_type : FrameType
        Frame type: "OLS", "LSS", "ESH", or "LPS"
    win_type : WindowType
        Window type: "KBD" or "SIN"

    Returns
    -------
    frame_T : np.ndarray
        Reconstructed frame in time domain, shape (2048, 2)

    Notes
    -----
    Processing steps:
    1. Apply IMDCT (produces N samples from N/2 coefficients)
    2. Apply same window used in forward transform
    3. Overlap-add will be performed in main encoder/decoder
    """
    # Step 1: Create appropriate window (same as forward)
    if frame_type == "ESH":
        from .constants import SHORT_FRAME_SIZE

        window = _create_window(SHORT_FRAME_SIZE, win_type, frame_type)
    else:
        from .constants import FRAME_SIZE

        window = _create_window(FRAME_SIZE, win_type, frame_type)

    # Step 2: Apply inverse MDCT and windowing for each channel
    from .constants import FRAME_SIZE

    frame_T = np.zeros((FRAME_SIZE, 2))

    if frame_type == "ESH":
        # Inverse process for 8 short subframes
        for ch in range(2):
            frame_T[:, ch] = _inverse_process_short_frame(frame_F[:, :, ch], window)
    else:
        # Inverse process for single long frame
        for ch in range(2):
            frame_T[:, ch] = _inverse_process_long_frame(frame_F[:, ch], window)

    return frame_T


def _create_window(N: int, win_type: WindowType, frame_type: FrameType) -> np.ndarray:
    """
    Create the appropriate window for a given frame type and window type.

    Parameters
    ----------
    N : int
        Window length (2048 for long, 256 for short)
    win_type : WindowType
        "KBD" or "SIN"
    frame_type : FrameType
        Frame type determines window shape

    Returns
    -------
    window : np.ndarray
        Window coefficients, shape (N,)

    Notes
    -----
    Window shapes (see Figure 4):
    - OLS: symmetric window of length N
    - LSS: left half of long window + flat + right half of short + zeros
    - ESH: symmetric short window
    - LPS: zeros + left half of short + flat + right half of long
    """
    # Select base window function
    create_fn = _create_kbd_window if win_type == "KBD" else _create_sin_window

    # OLS and ESH use symmetric windows
    if frame_type == "OLS" or frame_type == "ESH":
        return create_fn(N)

    # LSS and LPS need both long and short windows
    from .constants import FRAME_SIZE, SHORT_FRAME_SIZE

    win_long = create_fn(FRAME_SIZE)
    win_short = create_fn(SHORT_FRAME_SIZE)

    if frame_type == "LSS":
        return _create_long_start_window(win_long, win_short)
    elif frame_type == "LPS":
        return _create_long_stop_window(win_long, win_short)
    else:
        raise ValueError(f"Invalid frame type: {frame_type}")


def _create_kbd_window(N: int) -> np.ndarray:
    """
    Create Kaiser-Bessel-Derived (KBD) window.

    Parameters
    ----------
    N : int
        Window length (2048 or 256)

    Returns
    -------
    window : np.ndarray
        KBD window coefficients, shape (N,)

    Notes
    -----
    Uses alpha = 6 for long windows (N=2048)
    Uses alpha = 4 for short windows (N=256)

    Formula:
    Left half (n < N/2):
        W[n] = sqrt(sum(w[i], i=0..n) / sum(w[i], i=0..N/2))
    Right half (n >= N/2):
        W[n] = sqrt(sum(w[i], i=0..N-n) / sum(w[i], i=0..N/2))

    where w[n] is Kaiser window: use scipy.signal.windows.kaiser
    """
    import scipy.signal.windows

    # Select alpha based on window size
    alpha = 6.0 if N == 2048 else 4.0

    # Create Kaiser window of length N/2 + 1
    # beta = pi * alpha for scipy's kaiser
    beta = np.pi * alpha
    kaiser = scipy.signal.windows.kaiser(N // 2 + 1, beta)

    # Compute cumulative sum
    cumsum = np.cumsum(kaiser)

    # Normalize by total sum
    total = cumsum[-1]

    # Build KBD window
    window = np.zeros(N)

    # Left half: W[n] = sqrt(cumsum[n+1] / total)
    window[: N // 2] = np.sqrt(cumsum[:-1] / total)

    # Right half: symmetric (time-reversed left half)
    window[N // 2 :] = window[: N // 2][::-1]

    return window


def _create_sin_window(N: int) -> np.ndarray:
    """
    Create sinusoidal window.

    Parameters
    ----------
    N : int
        Window length

    Returns
    -------
    window : np.ndarray
        Sinusoidal window coefficients, shape (N,)

    Formula
    -------
    W[n] = sin(π/N * (n + 0.5)) for n = 0..N-1
    """
    n = np.arange(N)
    return np.sin(np.pi / N * (n + 0.5))


def _create_long_start_window(
    win_long: np.ndarray, win_short: np.ndarray
) -> np.ndarray:
    """
    Create LONG_START_SEQUENCE window by concatenating components.

    Concatenates:
    - Left half of long window (1024 samples)
    - 448 ones
    - Right half of short window (128 samples)
    - 448 zeros

    Parameters
    ----------
    win_long : np.ndarray
        Long window (2048,)
    win_short : np.ndarray
        Short window (256,)

    Returns
    -------
    window : np.ndarray
        LSS window, shape (2048,)
    """
    return np.concatenate(
        [
            win_long[:1024],  # Left half of long window
            np.ones(448),  # Flat portion
            win_short[128:],  # Right half of short window
            np.zeros(448),  # Zero portion
        ]
    )


def _create_long_stop_window(win_long: np.ndarray, win_short: np.ndarray) -> np.ndarray:
    """
    Create LONG_STOP_SEQUENCE window by concatenating components.

    Concatenates:
    - 448 zeros
    - Left half of short window (128 samples)
    - 448 ones
    - Right half of long window (1024 samples)

    Parameters
    ----------
    win_long : np.ndarray
        Long window (2048,)
    win_short : np.ndarray
        Short window (256,)

    Returns
    -------
    window : np.ndarray
        LPS window, shape (2048,)
    """
    return np.concatenate(
        [
            np.zeros(448),  # Zero portion
            win_short[:128],  # Left half of short window
            np.ones(448),  # Flat portion
            win_long[1024:],  # Right half of long window
        ]
    )


def _mdct(x: np.ndarray, N: int) -> np.ndarray:
    """
    Compute Modified Discrete Cosine Transform.

    Parameters
    ----------
    x : np.ndarray
        Input samples, shape (N,)
    N : int
        Transform size

    Returns
    -------
    X : np.ndarray
        MDCT coefficients, shape (N/2,)

    Formula
    -------
    X[k] = (2/N) * sum(x[n] * cos(2π/N * (n + n0) * (k + 0.5)))
    for k = 0..N/2-1, n = 0..N-1
    where n0 = (N/2 + 1) / 2
    """
    n0 = (N / 2 + 1) / 2
    n = np.arange(N)
    k = np.arange(N // 2)

    # Build cosine matrix: cos(2π/N * (n + n0) * (k + 0.5))
    # Shape: (N, N/2) where [n, k] = cos(...)
    cos_matrix = np.cos(2 * np.pi / N * np.outer(n + n0, k + 0.5))

    # X[k] = sum over n of x[n] * cos_matrix[n, k]
    # No scaling here - scaling is done in IMDCT for proper reconstruction
    X = x @ cos_matrix

    return X


def _imdct(X: np.ndarray, N: int) -> np.ndarray:
    """
    Compute Inverse Modified Discrete Cosine Transform.

    Parameters
    ----------
    X : np.ndarray
        MDCT coefficients, shape (N/2,)
    N : int
        Transform size

    Returns
    -------
    x : np.ndarray
        Reconstructed samples, shape (N,)

    Formula
    -------
    x[n] = (2/N) * sum(X[k] * cos(2π/N * (n + n0) * (k + 0.5)))
    for n = 0..N-1, k = 0..N/2-1
    where n0 = (N/2 + 1) / 2
    """
    n0 = (N / 2 + 1) / 2
    n = np.arange(N)
    k = np.arange(N // 2)

    # Build cosine matrix: cos(2π/N * (n + n0) * (k + 0.5))
    # Shape: (N, N/2) where [n, k] = cos(...)
    cos_matrix = np.cos(2 * np.pi / N * np.outer(n + n0, k + 0.5))

    # x[n] = sum over k of X[k] * cos_matrix[n, k]
    # Scale by 4/N for proper reconstruction with double windowing
    x = (4 / N) * (cos_matrix @ X)

    return x


def _apply_window(samples: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Apply window to samples (element-wise multiplication).

    Parameters
    ----------
    samples : np.ndarray
        Input samples
    window : np.ndarray
        Window coefficients (same length as samples)

    Returns
    -------
    windowed : np.ndarray
        Windowed samples
    """
    return samples * window


def _process_long_frame(
    frame: np.ndarray, window: np.ndarray, channel: int
) -> np.ndarray:
    """
    Process a long frame (OLS, LSS, or LPS) for a single channel.

    Parameters
    ----------
    frame : np.ndarray
        Time-domain frame, shape (2048, 2)
    window : np.ndarray
        Window to apply, shape (2048,)
    channel : int
        Channel index (0 or 1)

    Returns
    -------
    coeffs : np.ndarray
        MDCT coefficients, shape (1024,)
    """
    from .constants import FRAME_SIZE

    # Step 1: Extract single channel samples
    samples = frame[:, channel]

    # Step 2: Apply window
    windowed = _apply_window(samples, window)

    # Step 3: Apply MDCT transform
    coeffs = _mdct(windowed, FRAME_SIZE)

    return coeffs


def _process_short_frame(
    frame: np.ndarray, window: np.ndarray, channel: int
) -> np.ndarray:
    """
    Process EIGHT_SHORT_SEQUENCE frame for a single channel.

    Extracts 1152 central samples, divides into 8 overlapping subframes,
    applies windowing and MDCT to each.

    Parameters
    ----------
    frame : np.ndarray
        Time-domain frame, shape (2048, 2)
    window : np.ndarray
        Short window to apply, shape (256,)
    channel : int
        Channel index (0 or 1)

    Returns
    -------
    coeffs : np.ndarray
        MDCT coefficients, shape (128, 8)
        8 subframes, each with 128 coefficients
    """
    from .constants import (
        ESH_CENTRAL_SAMPLES,
        ESH_DISCARD_LEFT,
        SHORT_FRAME_SIZE,
        SHORT_HOP_SIZE,
    )

    # Step 1: Extract single channel and central 1152 samples
    # Discard 448 samples on left and right
    samples = frame[:, channel]
    central = samples[ESH_DISCARD_LEFT : ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES]

    # Step 2: Process 8 overlapping subframes (256 samples each, 50% overlap)
    coeffs = np.zeros((128, 8))
    for i in range(8):
        # Extract subframe with 50% overlap (hop = 128)
        start = i * SHORT_HOP_SIZE
        end = start + SHORT_FRAME_SIZE
        subframe = central[start:end]

        # Apply window
        windowed = _apply_window(subframe, window)

        # Apply MDCT
        coeffs[:, i] = _mdct(windowed, SHORT_FRAME_SIZE)

    return coeffs


def _inverse_process_long_frame(coeffs: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Inverse process for long frames.

    Parameters
    ----------
    coeffs : np.ndarray
        MDCT coefficients, shape (1024,)
    window : np.ndarray
        Window to apply, shape (2048,)

    Returns
    -------
    samples : np.ndarray
        Time-domain samples, shape (2048,)
    """
    from .constants import FRAME_SIZE

    # Step 1: Apply inverse MDCT
    samples = _imdct(coeffs, FRAME_SIZE)

    # Step 2: Apply window (same as forward)
    windowed = _apply_window(samples, window)

    return windowed


def _inverse_process_short_frame(coeffs: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Inverse process for EIGHT_SHORT_SEQUENCE frames.

    Parameters
    ----------
    coeffs : np.ndarray
        MDCT coefficients, shape (128, 8)
    window : np.ndarray
        Short window, shape (256,)

    Returns
    -------
    samples : np.ndarray
        Time-domain samples, shape (2048,)
        Reconstructed with proper positioning and zero-padding
    """
    from .constants import (
        ESH_CENTRAL_SAMPLES,
        ESH_DISCARD_LEFT,
        FRAME_SIZE,
        SHORT_FRAME_SIZE,
        SHORT_HOP_SIZE,
    )

    # Step 1: Initialize output with zeros (full frame size)
    samples = np.zeros(FRAME_SIZE)

    # Step 2: Reconstruct central 1152 samples from 8 subframes with overlap-add
    central = np.zeros(ESH_CENTRAL_SAMPLES)

    for i in range(8):
        # Inverse MDCT for this subframe
        subframe = _imdct(coeffs[:, i], SHORT_FRAME_SIZE)

        # Apply window
        windowed = _apply_window(subframe, window)

        # Overlap-add to central region
        start = i * SHORT_HOP_SIZE
        end = start + SHORT_FRAME_SIZE
        central[start:end] += windowed

    # Step 3: Place central samples in full frame (with zeros on sides)
    samples[ESH_DISCARD_LEFT : ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES] = central

    return samples
