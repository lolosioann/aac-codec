"""
Sequence Segmentation Control (SSC) Module.

Implements frame type detection for AAC encoding based on signal transient analysis.
"""

import numpy as np
import scipy.signal

from .types import FrameType


def SSC(
    frame_T: np.ndarray, next_frame_T: np.ndarray, prev_frame_type: FrameType
) -> FrameType:
    """
    Sequence Segmentation Control - Determines the frame type for AAC encoding.

    Analyzes the next frame to detect transients (sudden signal changes) and selects
    the appropriate frame type based on the previous frame type and transient detection.

    Frame types:
    - "OLS": ONLY_LONG_SEQUENCE - steady-state signal
    - "LSS": LONG_START_SEQUENCE - transition to transient
    - "ESH": EIGHT_SHORT_SEQUENCE - transient signal
    - "LPS": LONG_STOP_SEQUENCE - transition from transient

    Parameters
    ----------
    frame_T : np.ndarray
        Current frame in time domain, shape (2048, 2) for stereo channels
    next_frame_T : np.ndarray
        Next frame in time domain, shape (2048, 2), used for lookahead analysis
    prev_frame_type : FrameType
        Frame type selected for the previous frame (i-1)

    Returns
    -------
    frame_type : FrameType
        Selected frame type for the current frame (i)

    Notes
    -----
    State machine transitions:
    - OLS → OLS or LSS (LSS if next is transient)
    - LSS → ESH (forced)
    - ESH → ESH or LPS (ESH if next is transient)
    - LPS → OLS (forced)
    """
    # Step 1: Apply high-pass filter to next frame to emphasize transients
    filtered_next = _filter_next_frame(next_frame_T)

    # Step 2: Detect transients in each channel separately
    left_has_transient = _detect_transient_single_channel(filtered_next, channel=0)
    right_has_transient = _detect_transient_single_channel(filtered_next, channel=1)

    # Step 3: Apply state machine to each channel separately
    left_frame_type = _apply_state_machine(prev_frame_type, left_has_transient)
    right_frame_type = _apply_state_machine(prev_frame_type, right_has_transient)

    # Step 4: Combine per-channel frame types to get final frame type
    frame_type = _combine_channel_decisions(left_frame_type, right_frame_type)

    return frame_type


def _filter_next_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply high-pass filter to emphasize transients in the frame.

    Uses the filter: H(z) = (0.7548 - 0.7548*z^-1) / (1 - 0.5095*z^-1)

    Parameters
    ----------
    frame : np.ndarray
        Input frame, shape (2048, 2) for stereo

    Returns
    -------
    filtered : np.ndarray
        High-pass filtered frame, shape (2048, 2)

    Notes
    -----
    Use scipy.signal.lfilter for implementation
    """
    return scipy.signal.lfilter(
        b=[0.7548, -0.7548],
        a=[1, -0.5095],
        x=frame,
        axis=0,
    )


def _compute_segment_energies(filtered_frame: np.ndarray, channel: int) -> np.ndarray:
    """
    Compute energy (sum of squares) for 8 segments of 128 samples each.

    Parameters
    ----------
    filtered_frame : np.ndarray
        Filtered frame, shape (2048, 2)
    channel : int
        Channel index (0 or 1)

    Returns
    -------
    s2 : np.ndarray
        Energy values for 8 segments, shape (8,)
        s2[l] = sum(segment[l]^2) for l = 0..7
    """
    s2 = np.zeros(8)
    for section in range(8):
        segment = filtered_frame[
            448 + section * 128 : 448 + (section + 1) * 128, channel
        ]
        s2[section] = np.sum(segment**2)
    return s2


def _compute_attack_values(s2: np.ndarray) -> np.ndarray:
    """
    Compute attack values based on energy ratios.

    Attack value measures how much energy increases relative to previous segments.

    Parameters
    ----------
    s2 : np.ndarray
        Segment energies, shape (8,)

    Returns
    -------
    ds2 : np.ndarray
        Attack values, shape (8,)
        ds2[l] = s2[l] / mean(s2[0:l]) for l = 1..7
        ds2[0] = 0 (no previous segments)
    """
    # set first attack value to 0 (no previous segments)
    ds2 = np.zeros(8)
    for seg in range(1, 8):
        mean_prev = np.mean(s2[0:seg])
        if mean_prev > 0:
            ds2[seg] = s2[seg] / mean_prev
        else:
            ds2[seg] = 0.0  # Avoid division by zero
    return ds2


def _is_transient_segment(s2_l: float, ds2_l: float) -> bool:
    """
    Check if a segment satisfies transient conditions.

    Parameters
    ----------
    s2_l : float
        Energy of segment l
    ds2_l : float
        Attack value of segment l

    Returns
    -------
    is_transient : bool
        True if s2_l > 10^-3 AND ds2_l > 10
    """
    return (s2_l > 1e-3) and (ds2_l > 10)


def _detect_transient_single_channel(filtered_frame: np.ndarray, channel: int) -> bool:
    """
    Detect if a single channel contains transients.

    Parameters
    ----------
    filtered_frame : np.ndarray
        Filtered frame, shape (2048, 2)
    channel : int
        Channel index (0 or 1)

    Returns
    -------
    is_transient : bool
        True if any segment l=1..7 satisfies transient conditions
    """
    # Step 1: Compute energy for each of the 8 segments
    s2 = _compute_segment_energies(filtered_frame, channel)

    # Step 2: Compute attack values (energy ratio vs. previous segments)
    ds2 = _compute_attack_values(s2)

    # Step 3: Check if any segment l=1..7 satisfies transient conditions
    return any(
        _is_transient_segment(s2[idx], ds2[idx]) for idx in range(1, 8)
    )  # Start from 1, not 0 (need previous segments for attack)


def _combine_channel_decisions(
    left_frame_type: FrameType, right_frame_type: FrameType
) -> FrameType:
    """
    Combine left and right channel frame type decisions.

    Parameters
    ----------
    left_frame_type : FrameType
        Frame type determined for left channel
    right_frame_type : FrameType
        Frame type determined for right channel

    Returns
    -------
    frame_type : FrameType
        Combined frame type for the whole frame

    Notes
    -----
    Priority order (most restrictive wins):
    ESH > LSS > LPS > OLS

    This ensures transients are captured if either channel detects them.
    """
    priority = {"ESH": 0, "LSS": 1, "LPS": 2, "OLS": 3}

    if priority[left_frame_type] <= priority[right_frame_type]:
        return left_frame_type
    else:
        return right_frame_type


def _apply_state_machine(
    prev_frame_type: FrameType, next_has_transient: bool
) -> FrameType:
    """
    Apply state machine logic to determine current frame type.

    Parameters
    ----------
    prev_frame_type : FrameType
        Previous frame type
    next_has_transient : bool
        Whether next frame has transient detected

    Returns
    -------
    frame_type : FrameType
        Current frame type based on state machine rules

    Notes
    -----
    State transitions:
    - OLS → LSS if transient, else OLS
    - LSS → ESH (forced)
    - ESH → ESH if transient, else LPS
    - LPS → OLS (forced)
    """
    if prev_frame_type == "OLS":
        return "LSS" if next_has_transient else "OLS"
    elif prev_frame_type == "LSS":
        return "ESH"
    elif prev_frame_type == "ESH":
        return "ESH" if next_has_transient else "LPS"
    elif prev_frame_type == "LPS":
        return "OLS"
    else:
        raise ValueError(f"Invalid frame type: {prev_frame_type}")
