"""
Tests for AAC Coder Level 2 (with TNS).
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from part2.aac_coder_2 import aac_coder_2, i_aac_coder_2
from part2.constants import (
    HOP_SIZE,
    MDCT_LONG_SIZE,
    MDCT_SHORT_SIZE,
    NUM_SHORT_FRAMES,
    SAMPLE_RATE,
    TNS_ORDER,
)

from .conftest import compute_reconstruction_error

# Path to test audio file
TEST_WAV_FILE = "part2/input_stereo_38kHz.wav"


@pytest.fixture
def temp_wav_file() -> str:
    """Create a temporary stereo WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Generate short stereo sine wave
        duration = 0.5  # seconds
        t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right]).astype(np.float32)
        sf.write(f.name, stereo, SAMPLE_RATE)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_output_file() -> str:
    """Create a temporary output file path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


class TestAACCoder2:
    """Test AAC encoder Level 2."""

    def test_encodes_wav_file(self, temp_wav_file: str) -> None:
        """Should encode WAV file to sequence of frames."""
        aac_seq = aac_coder_2(temp_wav_file)
        assert isinstance(aac_seq, list)
        assert len(aac_seq) > 0

    def test_frame_sequence_structure(self, temp_wav_file: str) -> None:
        """Encoded sequence should have correct structure."""
        aac_seq = aac_coder_2(temp_wav_file)

        for frame in aac_seq:
            # Check required keys
            assert "frame_type" in frame
            assert "win_type" in frame
            assert "chl" in frame
            assert "chr" in frame

            # Check channel data structure
            assert "frame_F" in frame["chl"]
            assert "tns_coeffs" in frame["chl"]
            assert "frame_F" in frame["chr"]
            assert "tns_coeffs" in frame["chr"]

    def test_tns_coeffs_present(self, temp_wav_file: str) -> None:
        """Each frame should contain TNS coefficients."""
        aac_seq = aac_coder_2(temp_wav_file)

        for frame in aac_seq:
            chl_coeffs = frame["chl"]["tns_coeffs"]
            chr_coeffs = frame["chr"]["tns_coeffs"]

            if frame["frame_type"] == "ESH":
                assert chl_coeffs.shape == (TNS_ORDER, NUM_SHORT_FRAMES)
                assert chr_coeffs.shape == (TNS_ORDER, NUM_SHORT_FRAMES)
            else:
                assert chl_coeffs.shape == (TNS_ORDER, 1)
                assert chr_coeffs.shape == (TNS_ORDER, 1)

    def test_frame_types_valid(self, temp_wav_file: str) -> None:
        """Frame types should be valid."""
        aac_seq = aac_coder_2(temp_wav_file)
        valid_frame_types = {"OLS", "LSS", "ESH", "LPS"}

        for frame in aac_seq:
            assert frame["frame_type"] in valid_frame_types

    def test_mdct_coefficients_shape(self, temp_wav_file: str) -> None:
        """MDCT coefficients should have correct shape."""
        aac_seq = aac_coder_2(temp_wav_file)

        for frame in aac_seq:
            frame_type = frame["frame_type"]
            chl_F = frame["chl"]["frame_F"]
            chr_F = frame["chr"]["frame_F"]

            if frame_type == "ESH":
                assert chl_F.shape == (MDCT_SHORT_SIZE, NUM_SHORT_FRAMES)
                assert chr_F.shape == (MDCT_SHORT_SIZE, NUM_SHORT_FRAMES)
            else:
                assert chl_F.shape == (MDCT_LONG_SIZE, 1)
                assert chr_F.shape == (MDCT_LONG_SIZE, 1)


class TestIAACCoder2:
    """Test AAC decoder Level 2."""

    def test_decodes_sequence(self, temp_wav_file: str, temp_output_file: str) -> None:
        """Should decode sequence to audio signal."""
        aac_seq = aac_coder_2(temp_wav_file)
        decoded = i_aac_coder_2(aac_seq, temp_output_file)

        assert isinstance(decoded, np.ndarray)
        assert decoded.ndim == 2
        assert decoded.shape[1] == 2  # Stereo

    def test_output_shape(self, temp_wav_file: str, temp_output_file: str) -> None:
        """Output should be stereo with correct length."""
        aac_seq = aac_coder_2(temp_wav_file)
        decoded = i_aac_coder_2(aac_seq, temp_output_file)

        # Should have 2 channels
        assert decoded.shape[1] == 2

        # Length should be related to number of frames
        expected_min_samples = HOP_SIZE * (len(aac_seq) - 1)
        assert decoded.shape[0] >= expected_min_samples

    def test_saves_wav_file(self, temp_wav_file: str, temp_output_file: str) -> None:
        """Should save WAV file successfully."""
        aac_seq = aac_coder_2(temp_wav_file)
        i_aac_coder_2(aac_seq, temp_output_file)

        # Check file exists and is readable
        assert os.path.exists(temp_output_file)
        data, sr = sf.read(temp_output_file)
        assert sr == SAMPLE_RATE
        assert data.ndim == 2
        assert data.shape[1] == 2


class TestRoundtrip:
    """Test encode-decode roundtrip."""

    def test_roundtrip_reconstruction(
        self, temp_wav_file: str, temp_output_file: str
    ) -> None:
        """Encoding and decoding should reconstruct signal accurately."""
        # Read original
        original, sr = sf.read(temp_wav_file, dtype="float64")

        # Encode and decode
        aac_seq = aac_coder_2(temp_wav_file)
        decoded = i_aac_coder_2(aac_seq, temp_output_file)

        # Align lengths
        min_len = min(original.shape[0], decoded.shape[0])
        original_aligned = original[:min_len]
        decoded_aligned = decoded[:min_len]

        # Check correlation (should be high)
        for ch in range(2):
            corr = np.corrcoef(
                original_aligned[:, ch].flatten(), decoded_aligned[:, ch].flatten()
            )[0, 1]
            assert corr > 0.9  # High correlation expected

    def test_reconstruction_error_small(
        self, temp_wav_file: str, temp_output_file: str
    ) -> None:
        """Reconstruction error should be small."""
        # Read original
        original, sr = sf.read(temp_wav_file, dtype="float64")

        # Encode and decode
        aac_seq = aac_coder_2(temp_wav_file)
        decoded = i_aac_coder_2(aac_seq, temp_output_file)

        # Align lengths
        min_len = min(original.shape[0], decoded.shape[0])
        original_aligned = original[:min_len]
        decoded_aligned = decoded[:min_len]

        error = compute_reconstruction_error(original_aligned, decoded_aligned)
        # Short test signals have edge effects from overlap-add
        # Threshold is relaxed for synthetic test audio
        assert error < 0.2

    def test_snr_high(self, temp_wav_file: str, temp_output_file: str) -> None:
        """SNR should be reasonable for encoding."""
        # Read original
        original, sr = sf.read(temp_wav_file, dtype="float64")

        # Encode and decode
        aac_seq = aac_coder_2(temp_wav_file)
        decoded = i_aac_coder_2(aac_seq, temp_output_file)

        # Align lengths
        min_len = min(original.shape[0], decoded.shape[0])
        original_aligned = original[:min_len]
        decoded_aligned = decoded[:min_len]

        # Compute SNR
        signal_power = np.sum(original_aligned**2)
        noise_power = np.sum((original_aligned - decoded_aligned) ** 2)

        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            # Short generated signals have edge effects, threshold is relaxed
            # Real audio tests (TestWithRealAudio) verify high SNR (> 40 dB)
            assert snr_db > 15


class TestFrameTypeTransitions:
    """Test frame type state machine transitions."""

    def test_valid_transitions(self, temp_wav_file: str) -> None:
        """Frame type transitions should follow state machine rules."""
        aac_seq = aac_coder_2(temp_wav_file)

        valid_transitions = {
            "OLS": {"OLS", "LSS"},
            "LSS": {"ESH"},
            "ESH": {"ESH", "LPS"},
            "LPS": {"OLS"},
        }

        for i in range(1, len(aac_seq)):
            prev_type = aac_seq[i - 1]["frame_type"]
            curr_type = aac_seq[i]["frame_type"]
            assert (
                curr_type in valid_transitions[prev_type]
            ), f"Invalid transition: {prev_type} -> {curr_type}"


class TestWindowTypes:
    """Test window type handling."""

    def test_window_types_valid(self, temp_wav_file: str) -> None:
        """Window types should be valid."""
        aac_seq = aac_coder_2(temp_wav_file)
        valid_win_types = {"KBD", "SIN"}

        for frame in aac_seq:
            assert frame["win_type"] in valid_win_types


@pytest.mark.skipif(
    not os.path.exists(TEST_WAV_FILE), reason="Test audio file not found"
)
class TestWithRealAudio:
    """Test with real audio file."""

    def test_encode_real_audio(self) -> None:
        """Should encode real audio file."""
        aac_seq = aac_coder_2(TEST_WAV_FILE)
        assert len(aac_seq) > 0

    def test_roundtrip_real_audio(self) -> None:
        """Should achieve high SNR with real audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_file = f.name

        try:
            # Read original
            original, sr = sf.read(TEST_WAV_FILE, dtype="float64")

            # Encode and decode
            aac_seq = aac_coder_2(TEST_WAV_FILE)
            decoded = i_aac_coder_2(aac_seq, output_file)

            # Align lengths
            min_len = min(original.shape[0], decoded.shape[0])
            original_aligned = original[:min_len]
            decoded_aligned = decoded[:min_len]

            # Compute SNR
            signal_power = np.sum(original_aligned**2)
            noise_power = np.sum((original_aligned - decoded_aligned) ** 2)

            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                # Should achieve SNR similar to demo (> 40 dB)
                assert snr_db > 40
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
