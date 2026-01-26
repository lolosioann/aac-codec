"""
Psychoacoustic Model Module.

Implements perceptual audio coding model to compute masking thresholds
and Signal-to-Mask Ratios (SMR) for perceptually-guided quantization.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from .aac_types import FrameType
from .constants import EPS, NMT, TMN

# Load band tables at module level
_mat_data = loadmat("docs/TableB219.mat")
B219a: NDArray[np.float64] = _mat_data["B219a"]  # (69, 6) - long frames
B219b: NDArray[np.float64] = _mat_data["B219b"]  # (42, 6) - short frames

# Cache for spreading function matrices
_spreading_long: NDArray[np.float64] | None = None
_spreading_short: NDArray[np.float64] | None = None


# =============================================================================
# Phase 1: Utility Functions
# =============================================================================


def _apply_hann_window(frame: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply Hann window to time-domain frame.

    Parameters
    ----------
    frame : NDArray[np.float64]
        Time-domain samples, shape (2048,) or (256,)

    Returns
    -------
    sw : NDArray[np.float64]
        Windowed samples, same shape as input

    Notes
    -----
    Hann window: sw(n) = s(n) * (0.5 - 0.5*cos(pi*(n+0.5)/N))
    """
    N = len(frame)
    n = np.arange(N)
    window = 0.5 - 0.5 * np.cos(np.pi * (n + 0.5) / N)
    return frame * window


def _compute_fft_magnitude_phase(
    sw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute FFT magnitude and phase.

    Parameters
    ----------
    sw : NDArray[np.float64]
        Windowed signal, shape (2048,) or (256,)

    Returns
    -------
    r : NDArray[np.float64]
        Magnitude spectrum, shape (1024,) or (128,)
    f : NDArray[np.float64]
        Phase spectrum, shape (1024,) or (128,)

    Notes
    -----
    - Uses np.fft.fft()
    - Returns magnitude and phase for indices 0 to N/2-1
    """
    N = len(sw)
    fft_result = np.fft.fft(sw)
    # Take first N/2 bins (0 to N/2-1)
    fft_half = fft_result[: N // 2]
    r = np.abs(fft_half)
    f = np.angle(fft_half)
    return r, f


def _predict_magnitude_phase(
    r_1: NDArray[np.float64],
    r_2: NDArray[np.float64],
    f_1: NDArray[np.float64],
    f_2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Predict current magnitude and phase from two previous frames.

    Parameters
    ----------
    r_1, r_2 : NDArray[np.float64]
        Magnitude spectra of previous frames, shape (1024,) or (128,)
    f_1, f_2 : NDArray[np.float64]
        Phase spectra of previous frames, shape (1024,) or (128,)

    Returns
    -------
    rpred : NDArray[np.float64]
        Predicted magnitude
    fpred : NDArray[np.float64]
        Predicted phase

    Notes
    -----
    Linear prediction:
    - rpred(w) = 2*r_1(w) - r_2(w)
    - fpred(w) = 2*f_1(w) - f_2(w)
    """
    rpred = 2 * r_1 - r_2
    fpred = 2 * f_1 - f_2
    return rpred, fpred


def _compute_predictability(
    r: NDArray[np.float64],
    f: NDArray[np.float64],
    rpred: NDArray[np.float64],
    fpred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute predictability measure.

    Measures how well current spectrum can be predicted from past,
    indicating stationarity (tone-like vs noise-like).

    Parameters
    ----------
    r, f : NDArray[np.float64]
        Current magnitude and phase, shape (1024,) or (128,)
    rpred, fpred : NDArray[np.float64]
        Predicted magnitude and phase, same shape

    Returns
    -------
    c : NDArray[np.float64]
        Predictability values [0, 1], same shape as input

    Notes
    -----
    c(w) = sqrt((r*cos(f) - rpred*cos(fpred))^2 + (r*sin(f) - rpred*sin(fpred))^2)
           / (r + |rpred|)
    Lower c indicates more predictable (tone-like)
    """
    # Compute real and imaginary components
    real_diff = r * np.cos(f) - rpred * np.cos(fpred)
    imag_diff = r * np.sin(f) - rpred * np.sin(fpred)

    # Compute numerator (Euclidean distance)
    numerator = np.sqrt(real_diff**2 + imag_diff**2)

    # Compute denominator with EPS to avoid division by zero
    denominator = r + np.abs(rpred) + EPS

    c = numerator / denominator
    return c


# =============================================================================
# Phase 2: Band Processing
# =============================================================================


def _precompute_spreading_function(
    i: int,
    j: int,
    bval: NDArray[np.float64],
) -> float:
    """
    Compute spreading function value for band interaction.

    Implements the spreading function that models how masking energy
    from one critical band spreads to neighboring bands.

    Parameters
    ----------
    i : int
        Source band index (masking band)
    j : int
        Target band index (masked band)
    bval : NDArray[np.float64]
        Center frequencies of bands in Bark scale

    Returns
    -------
    x : float
        Spreading function value (energy transfer coefficient)

    Notes
    -----
    Implements PDF Section 2.4, Step 1 algorithm:
    - tmpx = 3.0*(bval[j] - bval[i]) if i >= j else 1.5*(bval[j] - bval[i])
    - tmpz = 8*min((tmpx - 0.5)^2 - 2*(tmpx - 0.5), 0)
    - tmpy = 15.811389 + 7.5*(tmpx + 0.474) - 17.5*sqrt(1.0 + (tmpx + 0.474)^2)
    - x = 10^((tmpz + tmpy)/10) if tmpy >= -100 else 0
    """
    tmpx = 3.0 * (bval[j] - bval[i]) if i >= j else 1.5 * (bval[j] - bval[i])

    tmpz = 8 * min((tmpx - 0.5) ** 2 - 2 * (tmpx - 0.5), 0)
    tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * np.sqrt(1.0 + (tmpx + 0.474) ** 2)

    if tmpy < -100:
        return 0.0
    return float(10 ** ((tmpz + tmpy) / 10))


def _get_spreading_matrix(band_table: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Get or compute the spreading function matrix for given band table.

    Parameters
    ----------
    band_table : NDArray[np.float64]
        Band definition table (B219a or B219b)

    Returns
    -------
    spreading_fn : NDArray[np.float64]
        Spreading function matrix, shape (NB, NB)
    """
    global _spreading_long, _spreading_short

    num_bands = band_table.shape[0]
    bval = band_table[:, 4]  # Column 4 is bval

    # Check cache
    if num_bands == 69 and _spreading_long is not None:
        return _spreading_long
    if num_bands == 42 and _spreading_short is not None:
        return _spreading_short

    # Compute spreading matrix
    spreading_fn = np.zeros((num_bands, num_bands))
    for i in range(num_bands):
        for j in range(num_bands):
            spreading_fn[i, j] = _precompute_spreading_function(i, j, bval)

    # Cache result
    if num_bands == 69:
        _spreading_long = spreading_fn
    else:
        _spreading_short = spreading_fn

    return spreading_fn


def _compute_band_energy(
    r: NDArray[np.float64],
    band_table: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute energy per psychoacoustic band.

    Parameters
    ----------
    r : NDArray[np.float64]
        Magnitude spectrum, shape (1024,) or (128,)
    band_table : NDArray[np.float64]
        Band definition with columns [index, w_low, w_high, width, bval, qsthr]

    Returns
    -------
    e : NDArray[np.float64]
        Energy per band, shape (69,) or (42,)

    Notes
    -----
    e(b) = sum_{w=w_low(b)}^{w_high(b)} r(w)^2
    """
    num_bands = band_table.shape[0]
    e = np.zeros(num_bands)

    for b in range(num_bands):
        w_low = int(band_table[b, 1])
        w_high = int(band_table[b, 2])
        e[b] = np.sum(r[w_low : w_high + 1] ** 2)

    return e


def _compute_weighted_predictability(
    c: NDArray[np.float64],
    r: NDArray[np.float64],
    band_table: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute energy-weighted predictability per band.

    Parameters
    ----------
    c : NDArray[np.float64]
        Predictability values, shape (1024,) or (128,)
    r : NDArray[np.float64]
        Magnitude spectrum, shape (1024,) or (128,)
    band_table : NDArray[np.float64]
        Band definition

    Returns
    -------
    c_band : NDArray[np.float64]
        Weighted predictability per band, shape (69,) or (42,)

    Notes
    -----
    c(b) = sum_{w=w_low(b)}^{w_high(b)} c(w)*r(w)^2
    """
    num_bands = band_table.shape[0]
    c_band = np.zeros(num_bands)

    for b in range(num_bands):
        w_low = int(band_table[b, 1])
        w_high = int(band_table[b, 2])
        c_band[b] = np.sum(c[w_low : w_high + 1] * r[w_low : w_high + 1] ** 2)

    return c_band


# =============================================================================
# Phase 3: Spreading & Normalization
# =============================================================================


def _apply_spreading(
    e: NDArray[np.float64],
    c: NDArray[np.float64],
    spreading_fn: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply spreading function to energy and predictability.

    Models how masking spreads across frequency bands.

    Parameters
    ----------
    e : NDArray[np.float64]
        Energy per band, shape (69,) or (42,)
    c : NDArray[np.float64]
        Weighted predictability per band, same shape
    spreading_fn : NDArray[np.float64]
        Precomputed spreading function matrix, shape (NB, NB)

    Returns
    -------
    ecb : NDArray[np.float64]
        Spread energy, shape (69,) or (42,)
    ct : NDArray[np.float64]
        Spread predictability, same shape

    Notes
    -----
    ecb(b) = sum_{bb=0}^{NB-1} e(bb)*spreading_function(bb, b)
    ct(b) = sum_{bb=0}^{NB-1} c(bb)*spreading_function(bb, b)
    """
    # Matrix multiplication: spreading_fn.T @ e gives ecb
    # spreading_fn[i,j] = spreading from band i to band j
    # ecb[b] = sum over bb: e[bb] * spreading_fn[bb, b]
    ecb = spreading_fn.T @ e
    ct = spreading_fn.T @ c
    return ecb, ct


def _normalize_spreading(
    ecb: NDArray[np.float64],
    ct: NDArray[np.float64],
    spreading_fn: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize spread energy and predictability.

    Parameters
    ----------
    ecb : NDArray[np.float64]
        Spread energy, shape (69,) or (42,)
    ct : NDArray[np.float64]
        Spread predictability, same shape
    spreading_fn : NDArray[np.float64]
        Spreading function matrix

    Returns
    -------
    cb : NDArray[np.float64]
        Normalized predictability
    en : NDArray[np.float64]
        Normalized energy

    Notes
    -----
    cb(b) = ct(b) / ecb(b)
    en(b) = ecb(b) / sum_{bb=0}^{NB-1} spreading_function(bb, b)
    """
    # cb(b) = ct(b) / ecb(b)
    cb = ct / (ecb + EPS)

    # en(b) = ecb(b) / sum over bb of spreading_fn[bb, b]
    spreading_sum = np.sum(spreading_fn, axis=0)  # sum over rows for each column
    en = ecb / (spreading_sum + EPS)

    return cb, en


# =============================================================================
# Phase 4: Tonality & Thresholds
# =============================================================================


def _compute_tonality_index(cb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute tonality index from normalized predictability.

    Indicates whether signal in each band is tone-like or noise-like.

    Parameters
    ----------
    cb : NDArray[np.float64]
        Normalized predictability, shape (69,) or (42,)

    Returns
    -------
    tb : NDArray[np.float64]
        Tonality index [0, 1], same shape
        - tb → 1: tone-like (predictable)
        - tb → 0: noise-like (unpredictable)

    Notes
    -----
    tb(b) = -0.299 - 0.43*ln(cb(b))
    Clamp to [0, 1] range
    """
    # Avoid log(0) by using EPS
    tb = -0.299 - 0.43 * np.log(cb + EPS)
    # Clamp to [0, 1]
    tb = np.clip(tb, 0.0, 1.0)
    return tb


def _compute_snr_requirement(
    tb: NDArray[np.float64],
    NMT: float,
    TMN: float,
) -> NDArray[np.float64]:
    """
    Compute required SNR based on tonality.

    Parameters
    ----------
    tb : NDArray[np.float64]
        Tonality index, shape (69,) or (42,)
    NMT : float
        Noise Masking Tone threshold (dB)
    TMN : float
        Tone Masking Noise threshold (dB)

    Returns
    -------
    SNR : NDArray[np.float64]
        Required SNR per band (dB), same shape

    Notes
    -----
    SNR(b) = tb(b)*TMN + (1 - tb(b))*NMT
    - High tonality → use TMN (18 dB)
    - Low tonality → use NMT (6 dB)
    """
    SNR = tb * TMN + (1 - tb) * NMT
    return SNR


def _convert_snr_to_ratio(SNR: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert SNR from dB to linear ratio.

    Parameters
    ----------
    SNR : NDArray[np.float64]
        SNR in dB, shape (69,) or (42,)

    Returns
    -------
    bc : NDArray[np.float64]
        SNR as power ratio, same shape

    Notes
    -----
    bc(b) = 10^(-SNR(b)/10)
    """
    bc = 10 ** (-SNR / 10)
    return bc


def _compute_noise_threshold(
    en: NDArray[np.float64],
    bc: NDArray[np.float64],
    qthr: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute perceptual noise threshold per band.

    Combines masking threshold with absolute threshold of hearing.

    Parameters
    ----------
    en : NDArray[np.float64]
        Normalized energy, shape (69,) or (42,)
    bc : NDArray[np.float64]
        SNR ratio, same shape
    qthr : NDArray[np.float64]
        Absolute threshold in quiet (pre-computed q_thr_hat), same shape

    Returns
    -------
    npart : NDArray[np.float64]
        Noise threshold per band, same shape

    Notes
    -----
    nb(b) = en(b) * bc(b)
    npart(b) = max(nb(b), q_thr_hat(b))
    """
    nb = en * bc
    npart = np.maximum(nb, qthr)
    return npart


def _compute_smr(
    e: NDArray[np.float64],
    npart: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute Signal-to-Mask Ratio.

    Parameters
    ----------
    e : NDArray[np.float64]
        Signal energy per band, shape (69,) or (42,)
    npart : NDArray[np.float64]
        Noise threshold per band, same shape

    Returns
    -------
    SMR : NDArray[np.float64]
        Signal-to-Mask Ratio, same shape

    Notes
    -----
    SMR(b) = e(b) / npart(b)
    """
    SMR = e / (npart + EPS)
    return SMR


def _compute_perceptual_threshold(
    SMR: NDArray[np.float64],
    X_mdct: NDArray[np.float64],
    band_table: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute perceptual threshold in MDCT domain.

    Converts SMR to actual threshold for quantizer.

    Parameters
    ----------
    SMR : NDArray[np.float64]
        Signal-to-Mask Ratio per band, shape (69,) or (42,)
    X_mdct : NDArray[np.float64]
        MDCT coefficients, shape (1024,) or (128,)
    band_table : NDArray[np.float64]
        Band definitions

    Returns
    -------
    T : NDArray[np.float64]
        Perceptual threshold per band, shape (69,) or (42,)

    Notes
    -----
    P(b) = sum_{k=w_low(b)}^{w_high(b)} X(k)^2
    T(b) = P(b) / SMR(b)
    """
    num_bands = band_table.shape[0]
    T = np.zeros(num_bands)

    for b in range(num_bands):
        w_low = int(band_table[b, 1])
        w_high = int(band_table[b, 2])
        P_b = np.sum(X_mdct[w_low : w_high + 1] ** 2)
        T[b] = P_b / (SMR[b] + EPS)

    return T


# =============================================================================
# Phase 5: Main Function
# =============================================================================


def _compute_quiet_threshold(
    band_table: NDArray[np.float64], N: int
) -> NDArray[np.float64]:
    """
    Compute absolute threshold in quiet (q_thr_hat) for each band.

    Parameters
    ----------
    band_table : NDArray[np.float64]
        Band definition table
    N : int
        FFT size (2048 for long, 256 for short)

    Returns
    -------
    qthr : NDArray[np.float64]
        Quiet threshold per band

    Notes
    -----
    q_thr_hat = eps * (N/2) * 10^(qsthr/10)
    """
    qsthr = band_table[:, 5]  # Column 5 is qsthr
    qthr = EPS * (N / 2) * (10 ** (qsthr / 10))
    return qthr


def _process_single_window(
    frame: NDArray[np.float64],
    r_prev1: NDArray[np.float64],
    f_prev1: NDArray[np.float64],
    r_prev2: NDArray[np.float64],
    f_prev2: NDArray[np.float64],
    band_table: NDArray[np.float64],
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Process a single window (frame or subframe) through psychoacoustic pipeline.

    Returns SMR and current r, f for use as previous in next iteration.
    """
    # Step 2: Apply Hann window and compute FFT
    sw = _apply_hann_window(frame)
    r, f = _compute_fft_magnitude_phase(sw)

    # Step 3: Predict magnitude/phase
    rpred, fpred = _predict_magnitude_phase(r_prev1, f_prev1, r_prev2, f_prev2)

    # Step 4: Compute predictability
    c = _compute_predictability(r, f, rpred, fpred)

    # Step 5: Compute band energy
    e = _compute_band_energy(r, band_table)

    # Step 6: Compute weighted predictability
    c_band = _compute_weighted_predictability(c, r, band_table)

    # Get spreading matrix
    spreading_fn = _get_spreading_matrix(band_table)

    # Step 7: Apply spreading
    ecb, ct = _apply_spreading(e, c_band, spreading_fn)

    # Step 8: Normalize
    cb, en = _normalize_spreading(ecb, ct, spreading_fn)

    # Step 9: Compute tonality index
    tb = _compute_tonality_index(cb)

    # Step 10: Compute SNR requirement
    SNR = _compute_snr_requirement(tb, NMT, TMN)

    # Step 11: Convert to ratio
    bc = _convert_snr_to_ratio(SNR)

    # Compute quiet threshold
    qthr = _compute_quiet_threshold(band_table, N)

    # Step 12: Compute noise threshold
    npart = _compute_noise_threshold(en, bc, qthr)

    # Step 13: Compute SMR
    SMR = _compute_smr(e, npart)

    return SMR, r, f


def _extract_subframes(frame: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """
    Extract 8 subframes of 256 samples from central 1152 samples.

    Parameters
    ----------
    frame : NDArray[np.float64]
        Full frame, shape (2048,)

    Returns
    -------
    subframes : list of NDArray[np.float64]
        8 subframes, each shape (256,)
    """
    subframes = []
    for i in range(8):
        start = 448 + i * 128
        end = start + 256
        subframes.append(frame[start:end])
    return subframes


def psycho(
    frame_T: NDArray[np.float64],
    frame_type: FrameType,
    frame_T_prev_1: NDArray[np.float64],
    frame_T_prev_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute Signal-to-Mask Ratio (SMR) using psychoacoustic model.

    Analyzes audio frame to determine perceptual masking thresholds,
    which guide the quantizer to shape noise below audibility.

    Parameters
    ----------
    frame_T : NDArray[np.float64]
        Current frame in time domain, shape (2048,) for single channel
    frame_type : FrameType
        Frame type: "OLS", "LSS", "ESH", or "LPS"
    frame_T_prev_1 : NDArray[np.float64]
        Previous frame (i-1), shape (2048,)
    frame_T_prev_2 : NDArray[np.float64]
        Pre-previous frame (i-2), shape (2048,)

    Returns
    -------
    SMR : NDArray[np.float64]
        Signal-to-Mask Ratio per band
        - Shape (69, 1) for long frames
        - Shape (42, 8) for short frames (ESH)

    Notes
    -----
    Processing steps (following PDF Section 2.4):
    1. Precompute spreading function
    2. Apply Hann window and compute FFT
    3. Predict magnitude/phase from previous frames
    4. Compute predictability
    5-6. Compute band energy and apply spreading
    7. Compute tonality index
    8-9. Compute SNR requirement and convert to ratio
    10-11. Compute noise threshold
    12-13. Compute SMR and perceptual threshold
    """
    frame_T = frame_T.flatten()
    frame_T_prev_1 = frame_T_prev_1.flatten()
    frame_T_prev_2 = frame_T_prev_2.flatten()

    if frame_type == "ESH":
        # Process 8 short subframes
        band_table = B219b
        N = 256

        # Extract subframes from all three frames
        subframes_curr = _extract_subframes(frame_T)
        subframes_prev1 = _extract_subframes(frame_T_prev_1)
        subframes_prev2 = _extract_subframes(frame_T_prev_2)

        # Initialize output
        SMR_all = np.zeros((42, 8))

        # We need to track r, f history for prediction
        # Build a list of all subframes in order:
        # prev2: indices 0-7 (subframes from frame i-2)
        # prev1: indices 8-15 (subframes from frame i-1)
        # curr: indices 16-23 (subframes from frame i)
        # For subframe j in curr (index 16+j), we need:
        #   prev_1 = subframe at index 16+j-1 = 15+j
        #   prev_2 = subframe at index 16+j-2 = 14+j

        # Pre-compute r, f for all subframes from prev2 and prev1
        r_history = []
        f_history = []

        # Process prev2 subframes (8 of them)
        for sf in subframes_prev2:
            sw = _apply_hann_window(sf)
            r, f = _compute_fft_magnitude_phase(sw)
            r_history.append(r)
            f_history.append(f)

        # Process prev1 subframes (8 of them)
        for sf in subframes_prev1:
            sw = _apply_hann_window(sf)
            r, f = _compute_fft_magnitude_phase(sw)
            r_history.append(r)
            f_history.append(f)

        # Now process current subframes
        for j in range(8):
            # Index of current subframe in the combined history would be 16+j
            # prev_1 index = 15+j (8 from prev2 + 7+j from prev1 or earlier in curr)
            # prev_2 index = 14+j

            # For j=0: prev_1 = index 15 (subframe 7 from prev1), prev_2 = index 14 (subframe 6 from prev1)
            # For j=1: prev_1 = index 16 (subframe 0 from curr, but we haven't computed it yet)
            # So we need to compute incrementally

            if j == 0:
                # prev_1 = subframe 7 from prev1 (index 15 = 8+7)
                # prev_2 = subframe 6 from prev1 (index 14 = 8+6)
                r_prev1 = r_history[8 + 7]
                f_prev1 = f_history[8 + 7]
                r_prev2 = r_history[8 + 6]
                f_prev2 = f_history[8 + 6]
            elif j == 1:
                # prev_1 = subframe 0 from curr (just computed)
                # prev_2 = subframe 7 from prev1
                r_prev1 = r_history[-1]  # last added (subframe 0 of curr)
                f_prev1 = f_history[-1]
                r_prev2 = r_history[8 + 7]
                f_prev2 = f_history[8 + 7]
            else:
                # prev_1 = subframe j-1 from curr (2nd to last in history)
                # prev_2 = subframe j-2 from curr (3rd to last in history)
                r_prev1 = r_history[-1]
                f_prev1 = f_history[-1]
                r_prev2 = r_history[-2]
                f_prev2 = f_history[-2]

            SMR, r_curr, f_curr = _process_single_window(
                subframes_curr[j],
                r_prev1,
                f_prev1,
                r_prev2,
                f_prev2,
                band_table,
                N,
            )
            SMR_all[:, j] = SMR

            # Add current r, f to history for next iteration
            r_history.append(r_curr)
            f_history.append(f_curr)

        return SMR_all

    else:
        # Long frame (OLS, LSS, LPS)
        band_table = B219a
        N = 2048

        # Compute r, f for previous frames
        sw_prev1 = _apply_hann_window(frame_T_prev_1)
        r_prev1, f_prev1 = _compute_fft_magnitude_phase(sw_prev1)

        sw_prev2 = _apply_hann_window(frame_T_prev_2)
        r_prev2, f_prev2 = _compute_fft_magnitude_phase(sw_prev2)

        # Process current frame
        SMR, _, _ = _process_single_window(
            frame_T,
            r_prev1,
            f_prev1,
            r_prev2,
            f_prev2,
            band_table,
            N,
        )

        return SMR.reshape(-1, 1)
