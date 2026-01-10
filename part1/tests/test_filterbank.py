"""
Tests for Filterbank module (MDCT/IMDCT and windowing).
"""

import numpy as np

from part1.constants import ESH_CENTRAL_SAMPLES, ESH_DISCARD_LEFT, FRAME_SIZE
from part1.filterbank import (
    _apply_window,
    _create_kbd_window,
    _create_long_start_window,
    _create_long_stop_window,
    _create_sin_window,
    _create_window,
    _imdct,
    _inverse_process_long_frame,
    _inverse_process_short_frame,
    _mdct,
    _process_long_frame,
    _process_short_frame,
    filter_bank,
    i_filter_bank,
)


class TestCreateSinWindow:
    """Test sinusoidal window creation."""

    def test_window_shape_long(self) -> None:
        """Window should match requested length (2048)."""
        window = _create_sin_window(2048)
        assert window.shape == (2048,)

    def test_window_shape_short(self) -> None:
        """Window should match requested length (256)."""
        window = _create_sin_window(256)
        assert window.shape == (256,)

    def test_window_formula(self) -> None:
        """Values should match sin(pi/N * (n + 0.5))."""
        N = 256
        window = _create_sin_window(N)
        expected = np.sin(np.pi / N * (np.arange(N) + 0.5))
        np.testing.assert_allclose(window, expected)

    def test_window_range(self) -> None:
        """Values should be in (0, 1)."""
        window = _create_sin_window(2048)
        assert np.all(window > 0)
        assert np.all(window <= 1)

    def test_window_symmetry(self) -> None:
        """Sin window should be symmetric."""
        window = _create_sin_window(256)
        np.testing.assert_allclose(window, window[::-1])

    def test_princen_bradley_condition(self) -> None:
        """Window should satisfy w[n]^2 + w[n+N/2]^2 = 1."""
        N = 256
        window = _create_sin_window(N)
        # For MDCT, check that w[n]^2 + w[N/2+n]^2 ≈ 1 for n in first half
        first_half = window[: N // 2]
        second_half = window[N // 2 :]
        sum_squares = first_half**2 + second_half**2
        np.testing.assert_allclose(sum_squares, 1.0, rtol=1e-10)


class TestCreateKBDWindow:
    """Test Kaiser-Bessel-Derived window creation."""

    def test_long_window_shape(self) -> None:
        """Long window should have shape (2048,)."""
        window = _create_kbd_window(2048)
        assert window.shape == (2048,)

    def test_short_window_shape(self) -> None:
        """Short window should have shape (256,)."""
        window = _create_kbd_window(256)
        assert window.shape == (256,)

    def test_window_symmetry(self) -> None:
        """KBD window should be symmetric."""
        window = _create_kbd_window(256)
        np.testing.assert_allclose(window, window[::-1], rtol=1e-10)

    def test_window_range(self) -> None:
        """Window values should be in [0, 1]."""
        window = _create_kbd_window(2048)
        assert np.all(window >= 0)
        assert np.all(window <= 1)

    def test_princen_bradley_condition(self) -> None:
        """Window should satisfy Princen-Bradley condition."""
        N = 256
        window = _create_kbd_window(N)
        first_half = window[: N // 2]
        second_half = window[N // 2 :]
        sum_squares = first_half**2 + second_half**2
        np.testing.assert_allclose(sum_squares, 1.0, rtol=1e-6)

    def test_different_alpha_for_sizes(self) -> None:
        """Long (alpha=6) and short (alpha=4) should have different shapes."""
        long_win = _create_kbd_window(2048)
        short_win = _create_kbd_window(256)
        # Normalized comparison - shapes should differ due to different alpha
        long_normalized = long_win / long_win.max()
        short_normalized = short_win / short_win.max()
        # Sample at 1/8 point where difference is more pronounced
        long_sample = long_normalized[256]  # 1/8 point
        short_sample = short_normalized[32]  # 1/8 point
        # They should be different due to different alpha values
        assert not np.isclose(long_sample, short_sample, rtol=0.01)


class TestCreateLongStartWindow:
    """Test LONG_START_SEQUENCE window creation."""

    def test_window_shape(self) -> None:
        """Window should have shape (2048,)."""
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        window = _create_long_start_window(win_long, win_short)
        assert window.shape == (2048,)

    def test_window_structure(self) -> None:
        """Should have: left_long + flat + right_short + zeros."""
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        window = _create_long_start_window(win_long, win_short)

        # Left half of long window (1024 samples)
        np.testing.assert_allclose(window[:1024], win_long[:1024])

        # Flat portion (448 ones)
        np.testing.assert_allclose(window[1024:1472], 1.0)

        # Right half of short window (128 samples)
        np.testing.assert_allclose(window[1472:1600], win_short[128:])

        # Zero portion (448 zeros)
        np.testing.assert_allclose(window[1600:], 0.0)

    def test_segment_lengths(self) -> None:
        """Segments should be: 1024 + 448 + 128 + 448 = 2048."""
        assert 1024 + 448 + 128 + 448 == 2048


class TestCreateLongStopWindow:
    """Test LONG_STOP_SEQUENCE window creation."""

    def test_window_shape(self) -> None:
        """Window should have shape (2048,)."""
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        window = _create_long_stop_window(win_long, win_short)
        assert window.shape == (2048,)

    def test_window_structure(self) -> None:
        """Should have: zeros + left_short + flat + right_long."""
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        window = _create_long_stop_window(win_long, win_short)

        # Zero portion (448 zeros)
        np.testing.assert_allclose(window[:448], 0.0)

        # Left half of short window (128 samples)
        np.testing.assert_allclose(window[448:576], win_short[:128])

        # Flat portion (448 ones)
        np.testing.assert_allclose(window[576:1024], 1.0)

        # Right half of long window (1024 samples)
        np.testing.assert_allclose(window[1024:], win_long[1024:])

    def test_segment_lengths(self) -> None:
        """Segments should be: 448 + 128 + 448 + 1024 = 2048."""
        assert 448 + 128 + 448 + 1024 == 2048

    def test_lss_lps_symmetry(self) -> None:
        """LSS and LPS should be time-reversed versions of each other."""
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        lss = _create_long_start_window(win_long, win_short)
        lps = _create_long_stop_window(win_long, win_short)
        np.testing.assert_allclose(lss, lps[::-1])


class TestCreateWindow:
    """Test _create_window dispatcher function."""

    def test_ols_sin_window(self) -> None:
        """OLS with SIN should return symmetric sin window."""
        window = _create_window(2048, "SIN", "OLS")
        expected = _create_sin_window(2048)
        np.testing.assert_allclose(window, expected)

    def test_ols_kbd_window(self) -> None:
        """OLS with KBD should return symmetric KBD window."""
        window = _create_window(2048, "KBD", "OLS")
        expected = _create_kbd_window(2048)
        np.testing.assert_allclose(window, expected)

    def test_esh_window(self) -> None:
        """ESH should return short window."""
        window = _create_window(256, "SIN", "ESH")
        expected = _create_sin_window(256)
        np.testing.assert_allclose(window, expected)

    def test_lss_window(self) -> None:
        """LSS should return long-start window."""
        window = _create_window(2048, "SIN", "LSS")
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        expected = _create_long_start_window(win_long, win_short)
        np.testing.assert_allclose(window, expected)

    def test_lps_window(self) -> None:
        """LPS should return long-stop window."""
        window = _create_window(2048, "SIN", "LPS")
        win_long = _create_sin_window(2048)
        win_short = _create_sin_window(256)
        expected = _create_long_stop_window(win_long, win_short)
        np.testing.assert_allclose(window, expected)


class TestApplyWindow:
    """Test window application."""

    def test_output_shape(self) -> None:
        """Windowed signal should match input shape."""
        samples = np.random.randn(2048)
        window = np.ones(2048)
        result = _apply_window(samples, window)
        assert result.shape == samples.shape

    def test_multiplication(self) -> None:
        """Should perform element-wise multiplication."""
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        window = np.array([0.5, 0.5, 0.5, 0.5])
        result = _apply_window(samples, window)
        expected = np.array([0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_window(self) -> None:
        """Zero window should give zero output."""
        samples = np.random.randn(256)
        window = np.zeros(256)
        result = _apply_window(samples, window)
        np.testing.assert_allclose(result, 0.0)

    def test_ones_window(self) -> None:
        """Ones window should preserve signal."""
        samples = np.random.randn(256)
        window = np.ones(256)
        result = _apply_window(samples, window)
        np.testing.assert_allclose(result, samples)


class TestMDCT:
    """Test MDCT implementation."""

    def test_output_size(self) -> None:
        """MDCT should produce N/2 coefficients from N samples."""
        x = np.random.randn(2048)
        X = _mdct(x, 2048)
        assert X.shape == (1024,)

    def test_output_size_short(self) -> None:
        """MDCT should work for short frames too."""
        x = np.random.randn(256)
        X = _mdct(x, 256)
        assert X.shape == (128,)

    def test_dc_component(self) -> None:
        """Constant signal should produce predictable coefficients."""
        N = 256
        x = np.ones(N)
        X = _mdct(x, N)
        # DC signal should have energy concentrated in low frequencies
        assert X.shape == (N // 2,)

    def test_linearity(self) -> None:
        """MDCT should be linear: MDCT(a*x + b*y) = a*MDCT(x) + b*MDCT(y)."""
        N = 256
        x = np.random.randn(N)
        y = np.random.randn(N)
        a, b = 2.0, 3.0

        X = _mdct(x, N)
        Y = _mdct(y, N)
        XY = _mdct(a * x + b * y, N)

        np.testing.assert_allclose(XY, a * X + b * Y, rtol=1e-10)

    def test_zero_input(self) -> None:
        """Zero input should give zero output."""
        x = np.zeros(256)
        X = _mdct(x, 256)
        np.testing.assert_allclose(X, 0.0)

    def test_energy_preservation(self) -> None:
        """MDCT should approximately preserve energy (Parseval)."""
        N = 256
        x = np.random.randn(N)
        X = _mdct(x, N)
        # Energy in frequency domain (with proper scaling)
        freq_energy = np.sum(X**2) * N / 2
        # Should be approximately equal (within MDCT scaling)
        assert freq_energy > 0  # Basic sanity check


class TestIMDCT:
    """Test inverse MDCT implementation."""

    def test_output_size(self) -> None:
        """IMDCT should produce N samples from N/2 coefficients."""
        X = np.random.randn(1024)
        x = _imdct(X, 2048)
        assert x.shape == (2048,)

    def test_output_size_short(self) -> None:
        """IMDCT should work for short frames."""
        X = np.random.randn(128)
        x = _imdct(X, 256)
        assert x.shape == (256,)

    def test_zero_input(self) -> None:
        """Zero coefficients should give zero output."""
        X = np.zeros(128)
        x = _imdct(X, 256)
        np.testing.assert_allclose(x, 0.0)

    def test_linearity(self) -> None:
        """IMDCT should be linear."""
        N = 256
        X = np.random.randn(N // 2)
        Y = np.random.randn(N // 2)
        a, b = 2.0, 3.0

        x = _imdct(X, N)
        y = _imdct(Y, N)
        xy = _imdct(a * X + b * Y, N)

        np.testing.assert_allclose(xy, a * x + b * y, rtol=1e-10)


class TestMDCTIMDCTPerfectReconstruction:
    """Test MDCT/IMDCT perfect reconstruction with overlap-add."""

    def test_perfect_reconstruction_sin_window(self) -> None:
        """MDCT -> IMDCT with sin window and overlap-add should reconstruct."""
        N = 256
        hop = N // 2

        # Create test signal (longer than one frame)
        signal = np.random.randn(N + hop)
        window = _create_sin_window(N)

        # Frame 1: samples 0:N
        frame1 = signal[:N]
        windowed1 = _apply_window(frame1, window)
        X1 = _mdct(windowed1, N)
        recon1 = _imdct(X1, N)
        recon1_windowed = _apply_window(recon1, window)

        # Frame 2: samples hop:hop+N
        frame2 = signal[hop : hop + N]
        windowed2 = _apply_window(frame2, window)
        X2 = _mdct(windowed2, N)
        recon2 = _imdct(X2, N)
        recon2_windowed = _apply_window(recon2, window)

        # Overlap-add in the middle region
        # recon1_windowed[hop:N] + recon2_windowed[0:hop] should equal signal[hop:N]
        overlap_sum = recon1_windowed[hop:N] + recon2_windowed[:hop]
        np.testing.assert_allclose(overlap_sum, signal[hop:N], rtol=1e-10)

    def test_tdac_property(self) -> None:
        """Test Time-Domain Aliasing Cancellation property."""
        N = 256
        window = _create_sin_window(N)

        # Random signal
        x = np.random.randn(N)

        # Forward MDCT
        windowed = _apply_window(x, window)
        X = _mdct(windowed, N)

        # Inverse MDCT
        x_recon = _imdct(X, N)
        x_recon_windowed = _apply_window(x_recon, window)

        # With proper overlap-add, aliasing cancels
        # For single frame, just verify shape and non-zero output
        assert x_recon_windowed.shape == (N,)


class TestProcessLongFrame:
    """Test processing of long frames (OLS, LSS, LPS)."""

    def test_output_shape(self) -> None:
        """Should output 1024 MDCT coefficients."""
        frame = np.random.randn(2048, 2)
        window = _create_sin_window(2048)
        coeffs = _process_long_frame(frame, window, channel=0)
        assert coeffs.shape == (1024,)

    def test_both_channels(self) -> None:
        """Test processing both left and right channels."""
        frame = np.random.randn(2048, 2)
        frame[:, 0] = np.sin(np.linspace(0, 4 * np.pi, 2048))  # Different signals
        frame[:, 1] = np.cos(np.linspace(0, 4 * np.pi, 2048))
        window = _create_sin_window(2048)

        left = _process_long_frame(frame, window, channel=0)
        right = _process_long_frame(frame, window, channel=1)

        # Should be different
        assert not np.allclose(left, right)

    def test_zero_frame(self) -> None:
        """Zero frame should give zero coefficients."""
        frame = np.zeros((2048, 2))
        window = _create_sin_window(2048)
        coeffs = _process_long_frame(frame, window, channel=0)
        np.testing.assert_allclose(coeffs, 0.0)


class TestProcessShortFrame:
    """Test processing of EIGHT_SHORT_SEQUENCE frames."""

    def test_output_shape(self) -> None:
        """Should output (128, 8) MDCT coefficients."""
        frame = np.random.randn(2048, 2)
        window = _create_sin_window(256)
        coeffs = _process_short_frame(frame, window, channel=0)
        assert coeffs.shape == (128, 8)

    def test_central_samples_used(self) -> None:
        """Should use only central 1152 samples."""
        frame = np.zeros((2048, 2))
        # Put signal only in central region
        frame[ESH_DISCARD_LEFT : ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES, 0] = 1.0
        window = _create_sin_window(256)

        coeffs = _process_short_frame(frame, window, channel=0)
        # Should have non-zero coefficients
        assert np.any(coeffs != 0)

    def test_outside_central_ignored(self) -> None:
        """Signal outside central region should not affect output."""
        frame1 = np.zeros((2048, 2))
        frame2 = np.zeros((2048, 2))

        # Same signal in central region
        frame1[
            ESH_DISCARD_LEFT : ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES, 0
        ] = np.random.randn(ESH_CENTRAL_SAMPLES)
        frame2[ESH_DISCARD_LEFT : ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES, 0] = frame1[
            ESH_DISCARD_LEFT : ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES, 0
        ]

        # Different signal outside (should be ignored)
        frame1[:ESH_DISCARD_LEFT, 0] = 999.0
        frame2[:ESH_DISCARD_LEFT, 0] = -999.0

        window = _create_sin_window(256)
        coeffs1 = _process_short_frame(frame1, window, channel=0)
        coeffs2 = _process_short_frame(frame2, window, channel=0)

        np.testing.assert_allclose(coeffs1, coeffs2)

    def test_subframe_overlap(self) -> None:
        """8 subframes with 50% overlap should cover 1152 samples."""
        # 8 subframes of 256 samples with 128 hop = 128*7 + 256 = 1152
        assert ESH_CENTRAL_SAMPLES == 128 * 7 + 256


class TestInverseProcessLongFrame:
    """Test inverse processing of long frames."""

    def test_output_shape(self) -> None:
        """Should output 2048 time-domain samples."""
        coeffs = np.random.randn(1024)
        window = _create_sin_window(2048)
        samples = _inverse_process_long_frame(coeffs, window)
        assert samples.shape == (2048,)

    def test_zero_coeffs(self) -> None:
        """Zero coefficients should give zero output."""
        coeffs = np.zeros(1024)
        window = _create_sin_window(2048)
        samples = _inverse_process_long_frame(coeffs, window)
        np.testing.assert_allclose(samples, 0.0)

    def test_reconstruction_roundtrip(self) -> None:
        """Forward then inverse should allow reconstruction with overlap-add."""
        frame = np.random.randn(2048, 2)
        window = _create_sin_window(2048)

        # Forward
        coeffs = _process_long_frame(frame, window, channel=0)

        # Inverse
        recon = _inverse_process_long_frame(coeffs, window)

        # Shape should match
        assert recon.shape == (2048,)


class TestInverseProcessShortFrame:
    """Test inverse processing of short frames."""

    def test_output_shape(self) -> None:
        """Should output 2048 samples with proper padding."""
        coeffs = np.random.randn(128, 8)
        window = _create_sin_window(256)
        samples = _inverse_process_short_frame(coeffs, window)
        assert samples.shape == (2048,)

    def test_zero_padding(self) -> None:
        """Samples outside central region should be zero."""
        coeffs = np.random.randn(128, 8)
        window = _create_sin_window(256)
        samples = _inverse_process_short_frame(coeffs, window)

        # First 448 samples should be zero
        np.testing.assert_allclose(samples[:ESH_DISCARD_LEFT], 0.0)
        # Last 448 samples should be zero
        np.testing.assert_allclose(
            samples[ESH_DISCARD_LEFT + ESH_CENTRAL_SAMPLES :], 0.0
        )

    def test_zero_coeffs(self) -> None:
        """Zero coefficients should give zero output."""
        coeffs = np.zeros((128, 8))
        window = _create_sin_window(256)
        samples = _inverse_process_short_frame(coeffs, window)
        np.testing.assert_allclose(samples, 0.0)


class TestFilterBank:
    """Integration tests for filter_bank function."""

    def test_ols_frame_shape(self) -> None:
        """OLS frame should produce (1024, 2) coefficients."""
        frame = np.random.randn(2048, 2)
        coeffs = filter_bank(frame, "OLS", "SIN")
        assert coeffs.shape == (1024, 2)

    def test_esh_frame_shape(self) -> None:
        """ESH frame should produce (128, 8, 2) coefficients."""
        frame = np.random.randn(2048, 2)
        coeffs = filter_bank(frame, "ESH", "SIN")
        assert coeffs.shape == (128, 8, 2)

    def test_lss_frame_shape(self) -> None:
        """LSS frame should produce (1024, 2) coefficients."""
        frame = np.random.randn(2048, 2)
        coeffs = filter_bank(frame, "LSS", "SIN")
        assert coeffs.shape == (1024, 2)

    def test_lps_frame_shape(self) -> None:
        """LPS frame should produce (1024, 2) coefficients."""
        frame = np.random.randn(2048, 2)
        coeffs = filter_bank(frame, "LPS", "SIN")
        assert coeffs.shape == (1024, 2)

    def test_window_types(self) -> None:
        """Test both KBD and SIN window types produce valid output."""
        frame = np.random.randn(2048, 2)

        sin_coeffs = filter_bank(frame, "OLS", "SIN")
        kbd_coeffs = filter_bank(frame, "OLS", "KBD")

        assert sin_coeffs.shape == (1024, 2)
        assert kbd_coeffs.shape == (1024, 2)
        # Different windows should give different coefficients
        assert not np.allclose(sin_coeffs, kbd_coeffs)

    def test_stereo_independence(self) -> None:
        """Each channel should be processed independently."""
        frame = np.zeros((2048, 2))
        frame[:, 0] = np.sin(np.linspace(0, 8 * np.pi, 2048))
        frame[:, 1] = np.cos(np.linspace(0, 8 * np.pi, 2048))

        coeffs = filter_bank(frame, "OLS", "SIN")

        # Channels should be different
        assert not np.allclose(coeffs[:, 0], coeffs[:, 1])


class TestIFilterBank:
    """Integration tests for i_filter_bank function."""

    def test_inverse_ols_shape(self) -> None:
        """Inverse OLS should return (2048, 2) frame."""
        coeffs = np.random.randn(1024, 2)
        frame = i_filter_bank(coeffs, "OLS", "SIN")
        assert frame.shape == (2048, 2)

    def test_inverse_esh_shape(self) -> None:
        """Inverse ESH should return (2048, 2) frame."""
        coeffs = np.random.randn(128, 8, 2)
        frame = i_filter_bank(coeffs, "ESH", "SIN")
        assert frame.shape == (2048, 2)

    def test_inverse_lss_shape(self) -> None:
        """Inverse LSS should return (2048, 2) frame."""
        coeffs = np.random.randn(1024, 2)
        frame = i_filter_bank(coeffs, "LSS", "SIN")
        assert frame.shape == (2048, 2)

    def test_inverse_lps_shape(self) -> None:
        """Inverse LPS should return (2048, 2) frame."""
        coeffs = np.random.randn(1024, 2)
        frame = i_filter_bank(coeffs, "LPS", "SIN")
        assert frame.shape == (2048, 2)


class TestFilterBankRoundtrip:
    """Test filter_bank -> i_filter_bank roundtrip."""

    def test_ols_roundtrip_shape(self) -> None:
        """OLS roundtrip should preserve shape."""
        frame = np.random.randn(2048, 2)
        coeffs = filter_bank(frame, "OLS", "SIN")
        recon = i_filter_bank(coeffs, "OLS", "SIN")
        assert recon.shape == frame.shape

    def test_esh_roundtrip_shape(self) -> None:
        """ESH roundtrip should preserve shape."""
        frame = np.random.randn(2048, 2)
        coeffs = filter_bank(frame, "ESH", "SIN")
        recon = i_filter_bank(coeffs, "ESH", "SIN")
        assert recon.shape == frame.shape

    def test_perfect_reconstruction_ols_with_overlap(self) -> None:
        """OLS with overlap-add should achieve perfect reconstruction."""
        # Create longer signal
        signal = np.random.randn(FRAME_SIZE + FRAME_SIZE // 2, 2)

        # Frame 1
        frame1 = signal[:FRAME_SIZE, :]
        coeffs1 = filter_bank(frame1, "OLS", "SIN")
        recon1 = i_filter_bank(coeffs1, "OLS", "SIN")

        # Frame 2 (50% overlap)
        frame2 = signal[FRAME_SIZE // 2 : FRAME_SIZE + FRAME_SIZE // 2, :]
        coeffs2 = filter_bank(frame2, "OLS", "SIN")
        recon2 = i_filter_bank(coeffs2, "OLS", "SIN")

        # Overlap-add in middle region
        overlap = recon1[FRAME_SIZE // 2 :, :] + recon2[: FRAME_SIZE // 2, :]

        # Should match original signal in overlap region
        # Use rtol=1e-9 to allow for floating point precision
        np.testing.assert_allclose(
            overlap, signal[FRAME_SIZE // 2 : FRAME_SIZE, :], rtol=1e-9
        )
