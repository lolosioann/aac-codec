"""
Pytest fixtures for Part 2 tests.

Extends Part 1 fixtures with TNS-specific test data.
"""

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def stereo_silence() -> NDArray[np.float64]:
    """
    Generate silent stereo frame.

    Returns
    -------
    frame : NDArray[np.float64]
        Silent frame, shape (2048, 2)
    """
    raise NotImplementedError()


@pytest.fixture
def stereo_sine_wave() -> NDArray[np.float64]:
    """
    Generate stereo sine wave frame (440 Hz).

    Returns
    -------
    frame : NDArray[np.float64]
        Sine wave frame, shape (2048, 2)
    """
    raise NotImplementedError()


@pytest.fixture
def stereo_impulse() -> NDArray[np.float64]:
    """
    Generate stereo impulse (transient).

    Returns
    -------
    frame : NDArray[np.float64]
        Impulse at center, shape (2048, 2)
    """
    raise NotImplementedError()


@pytest.fixture
def stereo_noise() -> NDArray[np.float64]:
    """
    Generate white noise stereo frame.

    Returns
    -------
    frame : NDArray[np.float64]
        Noise frame, shape (2048, 2)
    """
    raise NotImplementedError()


@pytest.fixture
def mdct_coeffs_long() -> NDArray[np.float64]:
    """
    Generate sample MDCT coefficients for long frame.

    Returns
    -------
    coeffs : NDArray[np.float64]
        MDCT coefficients, shape (1024,)
    """
    raise NotImplementedError()


@pytest.fixture
def mdct_coeffs_short() -> NDArray[np.float64]:
    """
    Generate sample MDCT coefficients for short frames.

    Returns
    -------
    coeffs : NDArray[np.float64]
        MDCT coefficients, shape (128, 8)
    """
    raise NotImplementedError()


@pytest.fixture
def tns_coeffs() -> NDArray[np.float64]:
    """
    Generate sample TNS coefficients.

    Returns
    -------
    coeffs : NDArray[np.float64]
        TNS filter coefficients, shape (4,)
    """
    raise NotImplementedError()


@pytest.fixture
def sample_aac_seq_2() -> list:
    """
    Generate sample AAC sequence for Part 2.

    Returns
    -------
    aac_seq : list[EncodedFrame2]
        Sample encoded sequence with a few frames
    """
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()
