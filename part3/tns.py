"""
Temporal Noise Shaping (TNS) Module.

Implements TNS processing to shape quantization noise in the temporal domain,
improving coding efficiency for transient signals.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from .aac_types import FrameType
from .constants import EPS, TNS_MAX_COEFF, TNS_ORDER, TNS_QUANTIZATION_STEP

# Load band tables at module level
_mat_data = loadmat("docs/TableB219.mat")
B219a: NDArray[np.float64] = _mat_data["B219a"]  # 69 bands for long frames
B219b: NDArray[np.float64] = _mat_data["B219b"]  # 42 bands for short frames


def tns(
    frame_F: NDArray[np.float64], frame_type: FrameType
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply Temporal Noise Shaping to MDCT coefficients for a single channel.

    TNS applies linear prediction in the frequency domain to shape quantization
    noise in the time domain, improving coding of transient signals.

    Parameters
    ----------
    frame_F : NDArray[np.float64]
        MDCT coefficients before TNS
        - Shape (1024,) or (1024, 1) for long frames (OLS, LSS, LPS)
        - Shape (128, 8) for short frames (ESH)
    frame_type : FrameType
        Frame type: "OLS", "LSS", "ESH", or "LPS"

    Returns
    -------
    frame_F_out : NDArray[np.float64]
        MDCT coefficients after TNS filtering
        - Same shape as input
    tns_coeffs : NDArray[np.float64]
        Quantized TNS filter coefficients
        - Shape (4, 1) for long frames
        - Shape (4, 8) for short frames

    Notes
    -----
    Processing steps:
    1. Normalize MDCT coefficients by band energy (Eqs 2-4)
    2. Smooth normalization coefficients
    3. Compute LPC coefficients (Eq 6)
    4. Quantize LPC coefficients (4-bit, step=0.1)
    5. Check filter stability
    6. Apply FIR TNS filter to original MDCT (Eq 8)
    """
    if frame_type == "ESH":
        # Process 8 short subframes independently
        frame_F_out = np.zeros_like(frame_F)
        tns_coeffs = np.zeros((TNS_ORDER, 8))

        for i in range(8):
            X = frame_F[:, i].flatten()
            X_tns, a_quant = _process_single_frame(X, B219b)
            frame_F_out[:, i] = X_tns
            tns_coeffs[:, i] = a_quant

        return frame_F_out, tns_coeffs
    else:
        # Long frame (OLS, LSS, LPS)
        X = frame_F.flatten()
        X_tns, a_quant = _process_single_frame(X, B219a)
        return X_tns.reshape(-1, 1), a_quant.reshape(-1, 1)


def _process_single_frame(
    X: NDArray[np.float64], band_table: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Process a single frame/subframe through TNS pipeline."""
    # Step 1: Normalize by band energy
    Xw, Sw = _normalize_mdct_coeffs(X, band_table)

    # Step 2: Smooth normalization coefficients
    Sw_smooth = _smooth_normalization(Sw)

    # Step 3: Apply smoothed normalization
    Xw_smooth = np.zeros_like(X)
    nonzero_mask = Sw_smooth > EPS
    Xw_smooth[nonzero_mask] = X[nonzero_mask] / Sw_smooth[nonzero_mask]

    # Step 4: Compute LPC coefficients
    a = _compute_lpc_coeffs(Xw_smooth, order=TNS_ORDER)

    # Step 5: Quantize coefficients
    a_quant = _quantize_tns_coeffs(a)

    # Step 6: Check stability and apply filter
    if _check_filter_stability(a_quant):
        X_tns = _apply_tns_filter(X, a_quant)
    else:
        # If unstable, bypass TNS (use zero coefficients)
        a_quant = np.zeros(TNS_ORDER)
        X_tns = X.copy()

    return X_tns, a_quant


def i_tns(
    frame_F: NDArray[np.float64],
    frame_type: FrameType,
    tns_coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply inverse TNS to recover original MDCT coefficients.

    Parameters
    ----------
    frame_F : NDArray[np.float64]
        TNS-filtered MDCT coefficients
        - Shape (1024,) or (1024, 1) for long frames
        - Shape (128, 8) for short frames
    frame_type : FrameType
        Frame type: "OLS", "LSS", "ESH", or "LPS"
    tns_coeffs : NDArray[np.float64]
        Quantized TNS filter coefficients
        - Shape (4, 1) for long frames
        - Shape (4, 8) for short frames

    Returns
    -------
    frame_F_out : NDArray[np.float64]
        Reconstructed MDCT coefficients (before TNS)
        - Same shape as input

    Notes
    -----
    Applies inverse IIR filter: H_TNS^(-1)(z)
    """
    if frame_type == "ESH":
        # Process 8 short subframes independently
        frame_F_out = np.zeros_like(frame_F)

        for i in range(8):
            X_tns = frame_F[:, i].flatten()
            a_quant = tns_coeffs[:, i].flatten()
            X = _apply_inverse_tns(X_tns, a_quant)
            frame_F_out[:, i] = X

        return frame_F_out
    else:
        # Long frame (OLS, LSS, LPS)
        X_tns = frame_F.flatten()
        a_quant = tns_coeffs.flatten()
        X = _apply_inverse_tns(X_tns, a_quant)
        return X.reshape(-1, 1)


def _normalize_mdct_coeffs(
    X: NDArray[np.float64],
    band_table: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize MDCT coefficients by band energy.

    Implements Equations 2-4 from the specification.

    Parameters
    ----------
    X : NDArray[np.float64]
        MDCT coefficients, shape (1024,) or (128,)
    band_table : NDArray[np.float64]
        Psychoacoustic band table with columns:
        [band_index, w_low, w_high, width, bval, qsthr]

    Returns
    -------
    Xw : NDArray[np.float64]
        Normalized MDCT coefficients, Xw(k) = X(k) / Sw(k)
    Sw : NDArray[np.float64]
        Normalization coefficients per coefficient, sqrt(P(j))

    Notes
    -----
    - P(j) = sum_{k=bj}^{bj+1-1} X(k)^2  (Eq 3)
    - Sw(k) = sqrt(P(j)) for bj <= k < bj+1  (Eq 4)
    - Sw will be smoothed before use in Eq 2
    """
    N = len(X)
    Sw = np.zeros(N)
    num_bands = band_table.shape[0]

    for j in range(num_bands):
        w_low = int(band_table[j, 1])
        w_high = int(band_table[j, 2])

        # Compute band energy P(j) = sum(X[k]^2) for k in [w_low, w_high]
        P_j = np.sum(X[w_low : w_high + 1] ** 2)

        # Sw(k) = sqrt(P(j)) for all k in this band
        Sw[w_low : w_high + 1] = np.sqrt(P_j + EPS)

    # Compute normalized coefficients
    Xw = np.zeros_like(X)
    nonzero_mask = Sw > EPS
    Xw[nonzero_mask] = X[nonzero_mask] / Sw[nonzero_mask]

    return Xw, Sw


def _smooth_normalization(Sw: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply double smoothing to normalization coefficients.

    Smooths abrupt changes between bands using forward and backward passes.

    Parameters
    ----------
    Sw : NDArray[np.float64]
        Normalization coefficients, shape (1024,) or (128,)

    Returns
    -------
    Sw_smooth : NDArray[np.float64]
        Smoothed normalization coefficients

    Notes
    -----
    Forward pass (k=N-2 down to 0): Sw(k) = (Sw(k) + Sw(k+1)) / 2
    Backward pass (k=1 to N-1): Sw(k) = (Sw(k) + Sw(k-1)) / 2
    """
    Sw_smooth = Sw.copy()
    N = len(Sw_smooth)

    # Forward pass: k = N-2 down to 0
    for k in range(N - 2, -1, -1):
        Sw_smooth[k] = (Sw_smooth[k] + Sw_smooth[k + 1]) / 2

    # Backward pass: k = 1 to N-1
    for k in range(1, N):
        Sw_smooth[k] = (Sw_smooth[k] + Sw_smooth[k - 1]) / 2

    return Sw_smooth


def _compute_lpc_coeffs(Xw: NDArray[np.float64], order: int = 4) -> NDArray[np.float64]:
    """
    Compute LPC coefficients using autocorrelation method.

    Solves normal equations Ra = r (Eq 6-7) to minimize prediction error (Eq 5).

    Parameters
    ----------
    Xw : NDArray[np.float64]
        Normalized MDCT coefficients, shape (1024,) or (128,)
    order : int, default=4
        LPC filter order (p)

    Returns
    -------
    a : NDArray[np.float64]
        LPC coefficients [a1, a2, ..., ap], shape (order,)

    Notes
    -----
    - Constructs autocorrelation matrix R and vector r
    - Solves Ra = r using Levinson-Durbin via solve_toeplitz
    - Returns unquantized coefficients
    """
    # Compute autocorrelation for lags 0 to order
    N = len(Xw)
    r = np.zeros(order + 1)

    for lag in range(order + 1):
        r[lag] = np.sum(Xw[: N - lag] * Xw[lag:N])

    # Handle edge case: zero signal
    if r[0] < EPS:
        return np.zeros(order)

    # Normalize autocorrelation
    r = r / (r[0] + EPS)

    # Solve Toeplitz system Ra = r[1:order+1]
    # R is Toeplitz with first row/column = r[0:order]
    # r_vec is r[1:order+1]
    try:
        a = solve_toeplitz(r[:order], r[1 : order + 1])
    except np.linalg.LinAlgError:
        # If singular, return zeros
        return np.zeros(order)

    return a


def _quantize_tns_coeffs(
    a: NDArray[np.float64],
    bits: int = 4,
    step: float = TNS_QUANTIZATION_STEP,
) -> NDArray[np.float64]:
    """
    Quantize TNS coefficients with uniform symmetric quantizer.

    Parameters
    ----------
    a : NDArray[np.float64]
        Unquantized LPC coefficients, shape (4,)
    bits : int, default=4
        Number of quantization bits
    step : float, default=0.1
        Quantization step size

    Returns
    -------
    a_quant : NDArray[np.float64]
        Quantized coefficients, shape (4,)

    Notes
    -----
    - Uses uniform quantizer with step size 0.1
    - 4-bit quantization gives 16 levels (-8 to +7 or -7 to +7)
    - Range: approximately [-0.8, 0.8]
    """
    # Quantize: round to nearest step
    a_quant = np.round(a / step) * step

    # Clip to valid range
    a_quant = np.clip(a_quant, -TNS_MAX_COEFF, TNS_MAX_COEFF)

    return a_quant


def _check_filter_stability(a_quant: NDArray[np.float64]) -> bool:
    """
    Check if inverse TNS filter H_TNS^(-1) is stable.

    A filter is stable if all poles are inside the unit circle |z| < 1.

    Parameters
    ----------
    a_quant : NDArray[np.float64]
        Quantized TNS coefficients, shape (4,)

    Returns
    -------
    is_stable : bool
        True if filter is causal and stable

    Notes
    -----
    - Computes poles of H_TNS^(-1)(z) = 1 / (1 - a1*z^-1 - ... - ap*z^-p)
    - The denominator polynomial is: z^p - a1*z^(p-1) - ... - ap
    - In standard form: [1, -a1, -a2, ..., -ap] for z^p, z^(p-1), ..., z^0
    - All poles must satisfy |pole| < 1
    """
    # If all coefficients are zero, filter is trivially stable
    if np.allclose(a_quant, 0):
        return True

    # Polynomial coefficients for denominator: 1 - a1*z^-1 - ... - ap*z^-p
    # Multiply by z^p: z^p - a1*z^(p-1) - ... - ap
    # Polynomial in descending powers: [1, -a1, -a2, -a3, -a4]
    # For numpy.polynomial.polynomial.Polynomial, use ascending powers: [-ap, ..., -a1, 1]
    p = len(a_quant)
    poly_coeffs = np.zeros(p + 1)
    poly_coeffs[0] = 1  # constant term (z^0 coeff when multiplied)
    poly_coeffs[1:] = -a_quant[::-1]  # ascending powers

    # Actually for the polynomial z^p - a1*z^(p-1) - ... - ap = 0
    # We need coefficients in ascending order: [-ap, -a(p-1), ..., -a1, 1]
    poly_coeffs_ascending = np.concatenate(
        [[-a_quant[p - 1 - i] for i in range(p)], [1]]
    )

    # Find roots
    poly = np.polynomial.polynomial.Polynomial(poly_coeffs_ascending)
    roots = poly.roots()

    # Check all roots are inside unit circle
    return np.all(np.abs(roots) < 1.0)


def _apply_tns_filter(
    X: NDArray[np.float64],
    a_quant: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply FIR TNS filter to MDCT coefficients.

    Implements Equation 8: H_TNS(z) = 1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p

    Parameters
    ----------
    X : NDArray[np.float64]
        Original MDCT coefficients, shape (1024,) or (128,)
    a_quant : NDArray[np.float64]
        Quantized TNS coefficients, shape (4,)

    Returns
    -------
    X_tns : NDArray[np.float64]
        TNS-filtered MDCT coefficients, same shape as input

    Notes
    -----
    - Uses quantized coefficients for reversibility
    - FIR filter: b = [1, -a1, -a2, -a3, -a4], a = [1]
    """
    # FIR filter coefficients: [1, -a1, -a2, -a3, -a4]
    b = np.concatenate([[1], -a_quant])
    a = np.array([1.0])

    # Apply FIR filter
    X_tns = lfilter(b, a, X)

    return X_tns


def _apply_inverse_tns(
    X_tns: NDArray[np.float64],
    a_quant: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply inverse IIR filter to recover original MDCT coefficients.

    Inverts the TNS filter: H_TNS^(-1)(z) = 1 / (1 - a1*z^-1 - ... - ap*z^-p)

    Parameters
    ----------
    X_tns : NDArray[np.float64]
        TNS-filtered MDCT coefficients, shape (1024,) or (128,)
    a_quant : NDArray[np.float64]
        Quantized TNS coefficients, shape (4,)

    Returns
    -------
    X : NDArray[np.float64]
        Recovered MDCT coefficients, same shape as input

    Notes
    -----
    - Applies IIR filter (inverse of FIR TNS filter)
    - IIR filter: b = [1], a = [1, -a1, -a2, -a3, -a4]
    - Should perfectly reconstruct if filter is stable
    """
    # IIR filter coefficients: b = [1], a = [1, -a1, -a2, -a3, -a4]
    b = np.array([1.0])
    a = np.concatenate([[1], -a_quant])

    # Apply IIR filter
    X = lfilter(b, a, X_tns)

    return X
