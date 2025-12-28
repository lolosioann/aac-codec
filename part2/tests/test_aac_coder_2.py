"""
Tests for AAC Coder Level 2 (with TNS).
"""


class TestAACCoder2:
    """Test AAC encoder Level 2."""

    def test_encodes_wav_file(self) -> None:
        """Should encode WAV file to sequence of frames."""
        pass

    def test_frame_sequence_structure(self) -> None:
        """Encoded sequence should have correct structure."""
        pass

    def test_tns_coeffs_present(self) -> None:
        """Each frame should contain TNS coefficients."""
        pass

    def test_frame_types_valid(self) -> None:
        """Frame types should be valid."""
        pass


class TestIAACCoder2:
    """Test AAC decoder Level 2."""

    def test_decodes_sequence(self) -> None:
        """Should decode sequence to audio signal."""
        pass

    def test_output_shape(self) -> None:
        """Output should be stereo with correct length."""
        pass

    def test_saves_wav_file(self) -> None:
        """Should save WAV file successfully."""
        pass


class TestRoundtrip:
    """Test encode-decode roundtrip."""

    def test_roundtrip_reconstruction(self) -> None:
        """Encoding and decoding should reconstruct signal accurately."""
        pass

    def test_reconstruction_error_small(self) -> None:
        """Reconstruction error should be small."""
        pass
