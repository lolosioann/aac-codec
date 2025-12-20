"""
Constants for AAC codec.

Contains all magic numbers, configuration values, and filter coefficients.
"""

import numpy as np

# Frame parameters
FRAME_SIZE = 2048  # samples per frame
HOP_SIZE = 1024  # 50% overlap
SHORT_FRAME_SIZE = 256  # for EIGHT_SHORT_SEQUENCE
SHORT_HOP_SIZE = 128  # 50% overlap for short frames
NUM_SHORT_FRAMES = 8  # number of short frames in ESH

# Sample rate
SAMPLE_RATE = 48000  # Hz

# MDCT output sizes
MDCT_LONG_SIZE = 1024  # N/2 coefficients from 2048 samples
MDCT_SHORT_SIZE = 128  # N/2 coefficients from 256 samples

# High-pass filter coefficients for transient detection
# H(z) = (0.7548 - 0.7548*z^-1) / (1 - 0.5095*z^-1)
HPF_B = np.array([0.7548, -0.7548])  # numerator
HPF_A = np.array([1.0, -0.5095])  # denominator

# Transient detection thresholds
ENERGY_THRESHOLD = 1e-3  # minimum energy for transient detection
ATTACK_THRESHOLD = 10.0  # minimum attack value for transient

# Segment parameters for transient detection
NUM_SEGMENTS = 8  # divide frame into 8 segments
SEGMENT_SIZE = 128  # samples per segment (256/2 for short frames)

# Window parameters
KBD_ALPHA_LONG = 6  # alpha for KBD window on long frames
KBD_ALPHA_SHORT = 4  # alpha for KBD window on short frames

# EIGHT_SHORT_SEQUENCE frame structure
ESH_CENTRAL_SAMPLES = 1152  # central samples used
ESH_DISCARD_LEFT = 448  # samples discarded on left
ESH_DISCARD_RIGHT = 448  # samples discarded on right

# LONG_START_SEQUENCE window structure
LSS_LEFT_LONG = 1024  # left half of long window
LSS_FLAT = 448  # flat portion (ones)
LSS_RIGHT_SHORT = 128  # right half of short window
LSS_ZEROS = 448  # zero portion

# LONG_STOP_SEQUENCE window structure
LPS_ZEROS = 448  # zero portion
LPS_LEFT_SHORT = 128  # left half of short window
LPS_FLAT = 448  # flat portion (ones)
LPS_RIGHT_LONG = 1024  # right half of long window

# MDCT parameters
MDCT_N0_LONG = (FRAME_SIZE // 2 + 1) / 2  # n0 for long frames
MDCT_N0_SHORT = (SHORT_FRAME_SIZE // 2 + 1) / 2  # n0 for short frames

# Numerical stability
EPS = np.finfo(float).eps  # machine epsilon

# Default configurations
DEFAULT_WINDOW_TYPE = "KBD"
DEFAULT_NUM_CHANNELS = 2  # stereo
