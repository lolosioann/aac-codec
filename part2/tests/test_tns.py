"""
Tests for Temporal Noise Shaping (TNS) module.
"""

import numpy as np
from numpy.typing import NDArray

from part2.constants import (
    MDCT_LONG_SIZE,
    MDCT_SHORT_SIZE,
    NUM_SHORT_FRAMES,
    TNS_MAX_COEFF,
    TNS_ORDER,
    TNS_QUANTIZATION_STEP,
)
from part2.tns import (
    B219a,
    B219b,
    _apply_inverse_tns,
    _apply_tns_filter,
    _check_filter_stability,
    _compute_lpc_coeffs,
    _normalize_mdct_coeffs,
    _quantize_tns_coeffs,
    _smooth_normalization,
    i_tns,
    tns,
)

from .conftest import assert_arrays_close, compute_reconstruction_error


class TestNormalizeMDCTCoeffs:
    """Test MDCT coefficient normalization by band energy."""

    def test_output_shapes(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Normalized coeffs and Sw should match input shape."""
        Xw, Sw = _normalize_mdct_coeffs(mdct_coeffs_long, B219a)
        assert Xw.shape == mdct_coeffs_long.shape
        assert Sw.shape == mdct_coeffs_long.shape

    def test_zero_input(self) -> None:
        """Zero input should handle gracefully."""
        X = np.zeros(MDCT_LONG_SIZE, dtype=np.float64)
        Xw, Sw = _normalize_mdct_coeffs(X, B219a)
        # Normalized coeffs should be zero for zero input
        assert np.allclose(Xw, 0)
        # Sw should be small epsilon values
        assert np.all(Sw >= 0)

    def test_normalization_values(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Normalization should produce non-zero Sw for non-zero input."""
        Xw, Sw = _normalize_mdct_coeffs(mdct_coeffs_long, B219a)
        # Sw should be positive where there's energy
        non_zero_bands = Sw > 0
        assert np.sum(non_zero_bands) > 0

    def test_short_frame(self, mdct_coeffs_short: NDArray[np.float64]) -> None:
        """Should work with short frame coefficients."""
        X = mdct_coeffs_short[:, 0]  # Take first subframe
        Xw, Sw = _normalize_mdct_coeffs(X, B219b)
        assert Xw.shape == X.shape
        assert Sw.shape == X.shape


class TestSmoothNormalization:
    """Test double smoothing of normalization coefficients."""

    def test_smoothing_reduces_discontinuities(self) -> None:
        """Smoothing should reduce abrupt changes between bands."""
        # Create step function
        Sw = np.zeros(MDCT_LONG_SIZE, dtype=np.float64)
        Sw[:512] = 1.0
        Sw[512:] = 10.0

        Sw_smooth = _smooth_normalization(Sw)

        # Check that transition is smoother
        # Gradient at boundary should be smaller after smoothing
        grad_original = np.abs(np.diff(Sw))
        grad_smooth = np.abs(np.diff(Sw_smooth))
        assert np.max(grad_smooth) < np.max(grad_original)

    def test_preserves_overall_scale(self) -> None:
        """Smoothing shouldn't drastically change overall magnitude."""
        rng = np.random.default_rng(42)
        Sw = np.abs(rng.standard_normal(MDCT_LONG_SIZE)) + 0.1

        Sw_smooth = _smooth_normalization(Sw)

        # Mean should be similar (within 50%)
        assert np.abs(np.mean(Sw_smooth) - np.mean(Sw)) / np.mean(Sw) < 0.5

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        Sw = np.ones(MDCT_LONG_SIZE, dtype=np.float64)
        Sw_smooth = _smooth_normalization(Sw)
        assert Sw_smooth.shape == Sw.shape


class TestComputeLPCCoeffs:
    """Test LPC coefficient computation."""

    def test_output_shape(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Should return p=4 coefficients."""
        a = _compute_lpc_coeffs(mdct_coeffs_long, order=TNS_ORDER)
        assert a.shape == (TNS_ORDER,)

    def test_white_noise(self) -> None:
        """White noise should give small LPC coefficients."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(MDCT_LONG_SIZE)
        a = _compute_lpc_coeffs(noise, order=TNS_ORDER)
        # White noise is uncorrelated, so LPC coeffs should be small
        assert np.max(np.abs(a)) < 0.5

    def test_periodic_signal(self) -> None:
        """Periodic signal should give significant LPC coefficients."""
        # Create periodic signal
        t = np.arange(MDCT_LONG_SIZE)
        periodic = np.sin(2 * np.pi * 10 * t / MDCT_LONG_SIZE)
        a = _compute_lpc_coeffs(periodic, order=TNS_ORDER)
        # Periodic signal is correlated, coeffs should be non-zero
        assert np.max(np.abs(a)) > 0.1

    def test_autocorrelation_method(
        self, mdct_coeffs_long: NDArray[np.float64]
    ) -> None:
        """LPC coefficients should be finite and bounded."""
        a = _compute_lpc_coeffs(mdct_coeffs_long, order=TNS_ORDER)
        assert np.all(np.isfinite(a))
        # Typical LPC coeffs are bounded
        assert np.all(np.abs(a) < 2.0)

    def test_zero_input(self) -> None:
        """Zero input should return zero coefficients."""
        X = np.zeros(MDCT_LONG_SIZE, dtype=np.float64)
        a = _compute_lpc_coeffs(X, order=TNS_ORDER)
        assert np.allclose(a, 0)


class TestQuantizeTNSCoeffs:
    """Test TNS coefficient quantization."""

    def test_quantization_step(self) -> None:
        """Quantized values should be multiples of step size."""
        a = np.array([0.123, -0.456, 0.789, -0.012])
        a_quant = _quantize_tns_coeffs(a)

        # Check each is multiple of step (within floating point tolerance)
        for coeff in a_quant:
            ratio = coeff / TNS_QUANTIZATION_STEP
            assert np.abs(ratio - np.round(ratio)) < 1e-10

    def test_range_clipping(self) -> None:
        """Values should be clipped to valid range."""
        a = np.array([1.5, -1.5, 2.0, -2.0])
        a_quant = _quantize_tns_coeffs(a)

        assert np.all(a_quant >= -TNS_MAX_COEFF)
        assert np.all(a_quant <= TNS_MAX_COEFF)

    def test_roundtrip(self) -> None:
        """Quantization should be deterministic."""
        a = np.array([0.15, -0.25, 0.35, -0.45])
        a_quant1 = _quantize_tns_coeffs(a)
        a_quant2 = _quantize_tns_coeffs(a)
        assert_arrays_close(a_quant1, a_quant2)

    def test_zero_input(self) -> None:
        """Zero input should give zero output."""
        a = np.zeros(TNS_ORDER, dtype=np.float64)
        a_quant = _quantize_tns_coeffs(a)
        assert np.allclose(a_quant, 0)


class TestFilterStability:
    """Test TNS filter stability check."""

    def test_stable_filter(self, stable_tns_coeffs: NDArray[np.float64]) -> None:
        """Stable filter should pass check."""
        assert _check_filter_stability(stable_tns_coeffs)

    def test_unstable_filter(self, unstable_tns_coeffs: NDArray[np.float64]) -> None:
        """Unstable filter should fail check."""
        assert not _check_filter_stability(unstable_tns_coeffs)

    def test_edge_cases(self) -> None:
        """Zero coefficients should be stable (trivial filter)."""
        a_zero = np.zeros(TNS_ORDER, dtype=np.float64)
        assert _check_filter_stability(a_zero)

    def test_small_coefficients_stable(self) -> None:
        """Small coefficients should be stable."""
        a_small = np.array([0.1, -0.1, 0.05, -0.05])
        assert _check_filter_stability(a_small)


class TestApplyTNSFilter:
    """Test forward TNS filtering."""

    def test_output_shape(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Output should match input shape."""
        a = np.array([0.2, -0.1, 0.1, 0.0])
        X_tns = _apply_tns_filter(mdct_coeffs_long, a)
        assert X_tns.shape == mdct_coeffs_long.shape

    def test_filter_effect(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Filter should change the signal (not identity)."""
        a = np.array([0.3, -0.2, 0.1, 0.0])
        X_tns = _apply_tns_filter(mdct_coeffs_long, a)
        # Should not be identical to input
        assert not np.allclose(X_tns, mdct_coeffs_long)

    def test_zero_coeffs_identity(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Zero coefficients should give identity (no filtering)."""
        a = np.zeros(TNS_ORDER, dtype=np.float64)
        X_tns = _apply_tns_filter(mdct_coeffs_long, a)
        assert_arrays_close(X_tns, mdct_coeffs_long)


class TestApplyInverseTNS:
    """Test inverse TNS filtering."""

    def test_output_shape(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Output should match input shape."""
        a = np.array([0.2, -0.1, 0.1, 0.0])
        X_tns = _apply_tns_filter(mdct_coeffs_long, a)
        X_recovered = _apply_inverse_tns(X_tns, a)
        assert X_recovered.shape == mdct_coeffs_long.shape

    def test_inverse_reconstruction(
        self, mdct_coeffs_long: NDArray[np.float64]
    ) -> None:
        """Inverse should recover original signal."""
        a = np.array([0.2, -0.1, 0.1, 0.0])  # Stable coefficients
        X_tns = _apply_tns_filter(mdct_coeffs_long, a)
        X_recovered = _apply_inverse_tns(X_tns, a)
        assert_arrays_close(X_recovered, mdct_coeffs_long, rtol=1e-10, atol=1e-10)

    def test_zero_coeffs_identity(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Zero coefficients should give identity."""
        a = np.zeros(TNS_ORDER, dtype=np.float64)
        X_recovered = _apply_inverse_tns(mdct_coeffs_long, a)
        assert_arrays_close(X_recovered, mdct_coeffs_long)


class TestTNS:
    """Integration tests for TNS forward processing."""

    def test_long_frame(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """TNS should process long frames correctly."""
        X = mdct_coeffs_long.reshape(-1, 1)
        X_tns, tns_coeffs = tns(X, "OLS")

        assert X_tns.shape == (MDCT_LONG_SIZE, 1)
        assert tns_coeffs.shape == (TNS_ORDER, 1)

    def test_short_frame(self, mdct_coeffs_short: NDArray[np.float64]) -> None:
        """TNS should process short frames (8 subframes) correctly."""
        X_tns, tns_coeffs = tns(mdct_coeffs_short, "ESH")

        assert X_tns.shape == (MDCT_SHORT_SIZE, NUM_SHORT_FRAMES)
        assert tns_coeffs.shape == (TNS_ORDER, NUM_SHORT_FRAMES)

    def test_roundtrip_accuracy(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """TNS + iTNS should reconstruct with high accuracy."""
        X = mdct_coeffs_long.reshape(-1, 1)
        X_tns, tns_coeffs = tns(X, "OLS")
        X_recovered = i_tns(X_tns, "OLS", tns_coeffs)

        error = compute_reconstruction_error(X, X_recovered)
        assert error < 1e-10  # Very small error expected

    def test_tns_coeffs_quantized(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """Returned TNS coeffs should be quantized."""
        X = mdct_coeffs_long.reshape(-1, 1)
        _, tns_coeffs = tns(X, "OLS")

        # Check quantization
        for coeff in tns_coeffs.flatten():
            if coeff != 0:
                ratio = coeff / TNS_QUANTIZATION_STEP
                assert np.abs(ratio - np.round(ratio)) < 1e-10

    def test_all_long_frame_types(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """TNS should work with all long frame types."""
        X = mdct_coeffs_long.reshape(-1, 1)
        for frame_type in ["OLS", "LSS", "LPS"]:
            X_tns, tns_coeffs = tns(X, frame_type)
            assert X_tns.shape == (MDCT_LONG_SIZE, 1)
            assert tns_coeffs.shape == (TNS_ORDER, 1)


class TestITNS:
    """Integration tests for inverse TNS."""

    def test_inverse_reconstruction(
        self, mdct_coeffs_long: NDArray[np.float64]
    ) -> None:
        """iTNS should perfectly reconstruct original MDCT coeffs."""
        X = mdct_coeffs_long.reshape(-1, 1)
        X_tns, tns_coeffs = tns(X, "OLS")
        X_recovered = i_tns(X_tns, "OLS", tns_coeffs)

        assert_arrays_close(X_recovered, X, rtol=1e-10, atol=1e-10)

    def test_long_frame(self, mdct_coeffs_long: NDArray[np.float64]) -> None:
        """iTNS should process long frames correctly."""
        X = mdct_coeffs_long.reshape(-1, 1)
        tns_coeffs = np.array([[0.2], [-0.1], [0.1], [0.0]])

        X_recovered = i_tns(X, "OLS", tns_coeffs)
        assert X_recovered.shape == (MDCT_LONG_SIZE, 1)

    def test_short_frame(self, mdct_coeffs_short: NDArray[np.float64]) -> None:
        """iTNS should process short frames (8 subframes) correctly."""
        X_tns, tns_coeffs = tns(mdct_coeffs_short, "ESH")
        X_recovered = i_tns(X_tns, "ESH", tns_coeffs)

        assert X_recovered.shape == (MDCT_SHORT_SIZE, NUM_SHORT_FRAMES)
        error = compute_reconstruction_error(mdct_coeffs_short, X_recovered)
        assert error < 1e-10


class TestTNSEdgeCases:
    """Edge case tests for TNS module."""

    def test_constant_signal(self) -> None:
        """Constant signal should be handled."""
        X = np.ones((MDCT_LONG_SIZE, 1), dtype=np.float64)
        X_tns, tns_coeffs = tns(X, "OLS")
        X_recovered = i_tns(X_tns, "OLS", tns_coeffs)
        assert_arrays_close(X_recovered, X, rtol=1e-10, atol=1e-10)

    def test_very_small_signal(self) -> None:
        """Very small signal should be handled without numerical issues."""
        X = np.ones((MDCT_LONG_SIZE, 1), dtype=np.float64) * 1e-10
        X_tns, tns_coeffs = tns(X, "OLS")
        X_recovered = i_tns(X_tns, "OLS", tns_coeffs)
        # Should recover or at least not produce NaN/Inf
        assert np.all(np.isfinite(X_recovered))

    def test_alternating_sign(self) -> None:
        """Alternating sign signal should be handled."""
        X = np.zeros((MDCT_LONG_SIZE, 1), dtype=np.float64)
        X[::2, 0] = 1.0
        X[1::2, 0] = -1.0
        X_tns, tns_coeffs = tns(X, "OLS")
        X_recovered = i_tns(X_tns, "OLS", tns_coeffs)
        assert_arrays_close(X_recovered, X, rtol=1e-10, atol=1e-10)
