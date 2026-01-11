"""
Pytest fixtures for Part 2 tests.

Extends Part 1 fixtures with TNS-specific test data.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from part2.constants import (
    FRAME_SIZE,
    MDCT_LONG_SIZE,
    MDCT_SHORT_SIZE,
    NUM_SHORT_FRAMES,
    SAMPLE_RATE,
    TNS_ORDER,
)


@pytest.fixture
def stereo_silence() -> NDArray[np.float64]:
    """
    Generate silent stereo frame.

    Returns
    -------
    frame : NDArray[np.float64]
        Silent frame, shape (2048, 2)
    """
    return np.zeros((FRAME_SIZE, 2), dtype=np.float64)


@pytest.fixture
def stereo_sine_wave() -> NDArray[np.float64]:
    """
    Generate stereo sine wave frame (440 Hz).

    Returns
    -------
    frame : NDArray[np.float64]
        Sine wave frame, shape (2048, 2)
    """
    t = np.arange(FRAME_SIZE) / SAMPLE_RATE
    freq = 440.0
    sine = 0.5 * np.sin(2 * np.pi * freq * t)
    frame = np.column_stack([sine, sine])
    return frame.astype(np.float64)


@pytest.fixture
def stereo_impulse() -> NDArray[np.float64]:
    """
    Generate stereo impulse (transient).

    Returns
    -------
    frame : NDArray[np.float64]
        Impulse at center, shape (2048, 2)
    """
    frame = np.zeros((FRAME_SIZE, 2), dtype=np.float64)
    center = FRAME_SIZE // 2
    frame[center, :] = 1.0
    return frame


@pytest.fixture
def stereo_noise() -> NDArray[np.float64]:
    """
    Generate white noise stereo frame.

    Returns
    -------
    frame : NDArray[np.float64]
        Noise frame, shape (2048, 2)
    """
    rng = np.random.default_rng(42)
    return rng.standard_normal((FRAME_SIZE, 2)).astype(np.float64) * 0.5


@pytest.fixture
def mdct_coeffs_long() -> NDArray[np.float64]:
    """
    Generate sample MDCT coefficients for long frame.

    Returns
    -------
    coeffs : NDArray[np.float64]
        MDCT coefficients, shape (1024,)
    """
    rng = np.random.default_rng(42)
    # Mimic typical MDCT spectrum: higher energy at lower frequencies
    coeffs = rng.standard_normal(MDCT_LONG_SIZE)
    # Apply decay to simulate typical audio spectrum
    decay = np.exp(-np.arange(MDCT_LONG_SIZE) / 200)
    return (coeffs * decay).astype(np.float64)


@pytest.fixture
def mdct_coeffs_short() -> NDArray[np.float64]:
    """
    Generate sample MDCT coefficients for short frames.

    Returns
    -------
    coeffs : NDArray[np.float64]
        MDCT coefficients, shape (128, 8)
    """
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal((MDCT_SHORT_SIZE, NUM_SHORT_FRAMES))
    # Apply decay to each subframe
    decay = np.exp(-np.arange(MDCT_SHORT_SIZE) / 30)
    return (coeffs * decay[:, np.newaxis]).astype(np.float64)


@pytest.fixture
def tns_coeffs() -> NDArray[np.float64]:
    """
    Generate sample TNS coefficients.

    Returns
    -------
    coeffs : NDArray[np.float64]
        TNS filter coefficients, shape (4,)
    """
    # Typical stable TNS coefficients (small values ensure stability)
    return np.array([0.3, -0.2, 0.1, 0.0], dtype=np.float64)


@pytest.fixture
def stable_tns_coeffs() -> NDArray[np.float64]:
    """
    Generate stable TNS coefficients.

    Returns
    -------
    coeffs : NDArray[np.float64]
        Stable TNS filter coefficients, shape (4,)
    """
    # Quantized to step size 0.1
    return np.array([0.2, -0.1, 0.1, 0.0], dtype=np.float64)


@pytest.fixture
def unstable_tns_coeffs() -> NDArray[np.float64]:
    """
    Generate unstable TNS coefficients.

    Returns
    -------
    coeffs : NDArray[np.float64]
        Unstable TNS filter coefficients, shape (4,)
    """
    # Large coefficients that lead to poles outside unit circle
    return np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float64)


@pytest.fixture
def sample_aac_seq_2() -> list:
    """
    Generate sample AAC sequence for Part 2.

    Returns
    -------
    aac_seq : list[EncodedFrame2]
        Sample encoded sequence with a few frames
    """
    from part2.aac_types import EncodedFrame2

    frames = []
    rng = np.random.default_rng(42)

    # Create 3 sample frames: OLS, LSS, ESH
    for frame_type in ["OLS", "LSS", "ESH"]:
        if frame_type == "ESH":
            chl_F = rng.standard_normal((MDCT_SHORT_SIZE, NUM_SHORT_FRAMES))
            chr_F = rng.standard_normal((MDCT_SHORT_SIZE, NUM_SHORT_FRAMES))
            chl_coeffs = np.zeros((TNS_ORDER, NUM_SHORT_FRAMES))
            chr_coeffs = np.zeros((TNS_ORDER, NUM_SHORT_FRAMES))
        else:
            chl_F = rng.standard_normal((MDCT_LONG_SIZE, 1))
            chr_F = rng.standard_normal((MDCT_LONG_SIZE, 1))
            chl_coeffs = np.zeros((TNS_ORDER, 1))
            chr_coeffs = np.zeros((TNS_ORDER, 1))

        frame: EncodedFrame2 = {
            "frame_type": frame_type,
            "win_type": "KBD",
            "chl": {
                "frame_F": chl_F,
                "tns_coeffs": chl_coeffs,
            },
            "chr": {
                "frame_F": chr_F,
                "tns_coeffs": chr_coeffs,
            },
        }
        frames.append(frame)

    return frames


def assert_arrays_close(
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert two arrays are close within tolerance.

    Parameters
    ----------
    actual : NDArray[np.float64]
        Actual array
    expected : NDArray[np.float64]
        Expected array
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    """
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def compute_reconstruction_error(
    original: NDArray[np.float64],
    reconstructed: NDArray[np.float64],
) -> float:
    """
    Compute normalized reconstruction error.

    Parameters
    ----------
    original : NDArray[np.float64]
        Original signal
    reconstructed : NDArray[np.float64]
        Reconstructed signal

    Returns
    -------
    error : float
        Normalized RMS error
    """
    diff = original - reconstructed
    rms_error = np.sqrt(np.mean(diff**2))
    rms_original = np.sqrt(np.mean(original**2))
    if rms_original < 1e-10:
        return rms_error
    return rms_error / rms_original
