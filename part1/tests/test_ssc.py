"""
Tests for Sequence Segmentation Control (SSC) module.
"""

import numpy as np
import pytest

from part1.ssc import (
    SSC,
    _apply_state_machine,
    _combine_channel_decisions,
    _compute_attack_values,
    _compute_segment_energies,
    _detect_transient_single_channel,
    _filter_next_frame,
    _is_transient_segment,
)


class TestFilterNextFrame:
    """Test high-pass filtering for transient detection."""

    def test_filter_output_shape(self, stereo_sine_wave: np.ndarray) -> None:
        """Filter should maintain input shape."""
        filtered = _filter_next_frame(stereo_sine_wave)
        assert filtered.shape == stereo_sine_wave.shape

    def test_filter_dc_removal(self) -> None:
        """High-pass filter should remove DC component."""
        # Constant DC signal
        dc_signal = np.ones((2048, 2)) * 0.5
        filtered = _filter_next_frame(dc_signal)

        # After settling, output should approach zero for DC input
        # Check last portion where filter has settled
        assert np.abs(filtered[-100:, :]).mean() < 0.01

    def test_filter_with_impulse(self) -> None:
        """Test filter response to impulse."""
        impulse = np.zeros((2048, 2))
        impulse[100, :] = 1.0

        filtered = _filter_next_frame(impulse)

        # Impulse should produce non-zero response
        assert np.abs(filtered).max() > 0
        # Output shape preserved
        assert filtered.shape == (2048, 2)

    def test_filter_preserves_high_frequency(self) -> None:
        """High-pass filter should preserve high frequency content."""
        # High frequency signal (close to Nyquist)
        t = np.arange(2048)
        high_freq = np.sin(2 * np.pi * 0.4 * t)  # 0.4 * fs/2
        signal = np.column_stack([high_freq, high_freq])

        filtered = _filter_next_frame(signal)

        # High frequency should pass through with some magnitude
        # Skip initial transient
        assert np.std(filtered[200:, 0]) > 0.1


class TestComputeSegmentEnergies:
    """Test energy calculation for 8 segments."""

    def test_energy_shape(self) -> None:
        """Should return 8 energy values."""
        frame = np.random.randn(2048, 2)
        s2 = _compute_segment_energies(frame, channel=0)
        assert s2.shape == (8,)

    def test_energy_values(self) -> None:
        """Energy should be sum of squares."""
        frame = np.zeros((2048, 2))
        # Put known values in first segment (samples 448:576)
        frame[448:576, 0] = 1.0  # 128 ones

        s2 = _compute_segment_energies(frame, channel=0)

        # First segment: sum of 128 ones squared = 128
        assert np.isclose(s2[0], 128.0)
        # Other segments should be zero
        assert np.allclose(s2[1:], 0.0)

    def test_zero_signal(self, stereo_silence: np.ndarray) -> None:
        """Zero signal should give zero energy."""
        s2 = _compute_segment_energies(stereo_silence, channel=0)
        assert np.allclose(s2, 0.0)

    def test_constant_signal(self) -> None:
        """Constant signal energy should match expected value."""
        frame = np.ones((2048, 2)) * 0.5

        s2 = _compute_segment_energies(frame, channel=0)

        # Each segment: 128 samples of 0.5^2 = 128 * 0.25 = 32
        expected = 128 * 0.25
        assert np.allclose(s2, expected)

    def test_channel_independence(self) -> None:
        """Each channel should be computed independently."""
        frame = np.zeros((2048, 2))
        frame[448:576, 0] = 1.0  # Left channel only
        frame[448:576, 1] = 2.0  # Right channel different

        s2_left = _compute_segment_energies(frame, channel=0)
        s2_right = _compute_segment_energies(frame, channel=1)

        assert np.isclose(s2_left[0], 128.0)
        assert np.isclose(s2_right[0], 128.0 * 4)  # 2^2 = 4


class TestComputeAttackValues:
    """Test attack value computation."""

    def test_attack_shape(self) -> None:
        """Should return 8 attack values."""
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        ds2 = _compute_attack_values(s2)
        assert ds2.shape == (8,)

    def test_first_attack_zero(self) -> None:
        """First attack value should be 0 (no previous segments)."""
        s2 = np.array([100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        ds2 = _compute_attack_values(s2)
        assert ds2[0] == 0.0

    def test_increasing_energy(self) -> None:
        """Increasing energy should give high attack values."""
        # Sudden jump: low energy then high
        s2 = np.array([0.001, 0.001, 0.001, 0.001, 100.0, 100.0, 100.0, 100.0])
        ds2 = _compute_attack_values(s2)

        # Segment 4 has huge jump relative to mean of previous
        # ds2[4] = 100 / mean([0.001, 0.001, 0.001, 0.001]) = 100 / 0.001 = 100000
        assert ds2[4] > 1000

    def test_constant_energy(self) -> None:
        """Constant energy should give attack value close to 1."""
        s2 = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        ds2 = _compute_attack_values(s2)

        # ds2[i] = s2[i] / mean(s2[0:i])
        # For constant: ds2[i] = 10 / 10 = 1
        assert np.allclose(ds2[1:], 1.0)

    def test_zero_previous_energy(self) -> None:
        """Should handle zero previous energy without division error."""
        s2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ds2 = _compute_attack_values(s2)

        # Should not raise, and zero division case returns 0
        assert ds2[1] == 0.0
        assert ds2[2] == 0.0
        assert ds2[3] == 0.0  # 1.0 / mean([0,0,0]) -> 0 (protected)


class TestIsTransientSegment:
    """Test transient detection for single segment."""

    def test_both_conditions_true(self) -> None:
        """Should return True when both conditions met."""
        # s2 > 1e-3 AND ds2 > 10
        assert _is_transient_segment(s2_l=0.1, ds2_l=15.0) is True

    def test_energy_too_low(self) -> None:
        """Should return False if energy below threshold."""
        # s2 <= 1e-3
        assert _is_transient_segment(s2_l=1e-4, ds2_l=100.0) is False

    def test_attack_too_low(self) -> None:
        """Should return False if attack below threshold."""
        # ds2 <= 10
        assert _is_transient_segment(s2_l=1.0, ds2_l=5.0) is False

    def test_both_conditions_false(self) -> None:
        """Should return False when both conditions fail."""
        assert _is_transient_segment(s2_l=1e-5, ds2_l=1.0) is False

    def test_boundary_energy(self) -> None:
        """Test energy at exact threshold."""
        # Exactly at threshold should be False (> not >=)
        assert _is_transient_segment(s2_l=1e-3, ds2_l=100.0) is False
        assert _is_transient_segment(s2_l=1.001e-3, ds2_l=100.0) is True

    def test_boundary_attack(self) -> None:
        """Test attack at exact threshold."""
        # Exactly at threshold should be False (> not >=)
        assert _is_transient_segment(s2_l=1.0, ds2_l=10.0) is False
        assert _is_transient_segment(s2_l=1.0, ds2_l=10.001) is True


class TestDetectTransientSingleChannel:
    """Test transient detection for one channel."""

    def test_no_transient(self, stereo_sine_wave: np.ndarray) -> None:
        """Steady signal should not trigger transient."""
        # Sine wave is steady state, no transient
        result = _detect_transient_single_channel(stereo_sine_wave, channel=0)
        assert result is False

    def test_with_transient(self) -> None:
        """Signal with attack should trigger transient."""
        # Create signal with sudden onset in detection region
        frame = np.zeros((2048, 2))
        # Detection region is samples 448:1472 (8 segments of 128)
        # Put silence in early segments, loud signal in later segment
        frame[448 : 448 + 128 * 3, 0] = 0.001  # Low energy first 3 segments
        frame[448 + 128 * 3 : 448 + 128 * 4, 0] = 1.0  # High energy segment 4

        result = _detect_transient_single_channel(frame, channel=0)
        assert result is True

    def test_silence_no_transient(self, stereo_silence: np.ndarray) -> None:
        """Silent signal should not trigger transient."""
        result = _detect_transient_single_channel(stereo_silence, channel=0)
        assert result is False

    def test_constant_no_transient(self) -> None:
        """Constant signal should not trigger transient."""
        frame = np.ones((2048, 2)) * 0.5
        result = _detect_transient_single_channel(frame, channel=0)
        assert result is False

    def test_channel_parameter(self) -> None:
        """Should detect transient only in specified channel."""
        frame = np.zeros((2048, 2))
        # Transient only in right channel
        frame[448 : 448 + 128 * 3, 1] = 0.001
        frame[448 + 128 * 3 : 448 + 128 * 4, 1] = 1.0

        assert _detect_transient_single_channel(frame, channel=0) is False
        assert _detect_transient_single_channel(frame, channel=1) is True


class TestCombineChannelDecisions:
    """Test combining left/right channel frame type decisions."""

    def test_both_ols(self) -> None:
        """Both OLS -> OLS."""
        assert _combine_channel_decisions("OLS", "OLS") == "OLS"

    def test_both_esh(self) -> None:
        """Both ESH -> ESH."""
        assert _combine_channel_decisions("ESH", "ESH") == "ESH"

    def test_esh_wins_over_ols(self) -> None:
        """ESH has priority over OLS."""
        assert _combine_channel_decisions("ESH", "OLS") == "ESH"
        assert _combine_channel_decisions("OLS", "ESH") == "ESH"

    def test_esh_wins_over_lss(self) -> None:
        """ESH has priority over LSS."""
        assert _combine_channel_decisions("ESH", "LSS") == "ESH"
        assert _combine_channel_decisions("LSS", "ESH") == "ESH"

    def test_esh_wins_over_lps(self) -> None:
        """ESH has priority over LPS."""
        assert _combine_channel_decisions("ESH", "LPS") == "ESH"
        assert _combine_channel_decisions("LPS", "ESH") == "ESH"

    def test_lss_wins_over_ols(self) -> None:
        """LSS has priority over OLS."""
        assert _combine_channel_decisions("LSS", "OLS") == "LSS"
        assert _combine_channel_decisions("OLS", "LSS") == "LSS"

    def test_lss_wins_over_lps(self) -> None:
        """LSS has priority over LPS."""
        assert _combine_channel_decisions("LSS", "LPS") == "LSS"
        assert _combine_channel_decisions("LPS", "LSS") == "LSS"

    def test_lps_wins_over_ols(self) -> None:
        """LPS has priority over OLS."""
        assert _combine_channel_decisions("LPS", "OLS") == "LPS"
        assert _combine_channel_decisions("OLS", "LPS") == "LPS"

    def test_priority_order(self) -> None:
        """Verify full priority order: ESH > LSS > LPS > OLS."""
        types = ["OLS", "LPS", "LSS", "ESH"]
        for i, higher in enumerate(types):
            for lower in types[:i]:
                assert _combine_channel_decisions(higher, lower) == higher
                assert _combine_channel_decisions(lower, higher) == higher


class TestApplyStateMachine:
    """Test state machine logic."""

    def test_ols_to_ols(self) -> None:
        """OLS -> OLS when next has no transient."""
        assert _apply_state_machine("OLS", next_has_transient=False) == "OLS"

    def test_ols_to_lss(self) -> None:
        """OLS -> LSS when next has transient."""
        assert _apply_state_machine("OLS", next_has_transient=True) == "LSS"

    def test_lss_to_esh_no_transient(self) -> None:
        """LSS -> ESH (forced, regardless of transient)."""
        assert _apply_state_machine("LSS", next_has_transient=False) == "ESH"

    def test_lss_to_esh_with_transient(self) -> None:
        """LSS -> ESH (forced, regardless of transient)."""
        assert _apply_state_machine("LSS", next_has_transient=True) == "ESH"

    def test_esh_to_esh(self) -> None:
        """ESH -> ESH when next has transient."""
        assert _apply_state_machine("ESH", next_has_transient=True) == "ESH"

    def test_esh_to_lps(self) -> None:
        """ESH -> LPS when next has no transient."""
        assert _apply_state_machine("ESH", next_has_transient=False) == "LPS"

    def test_lps_to_ols_no_transient(self) -> None:
        """LPS -> OLS (forced, regardless of transient)."""
        assert _apply_state_machine("LPS", next_has_transient=False) == "OLS"

    def test_lps_to_ols_with_transient(self) -> None:
        """LPS -> OLS (forced, regardless of transient)."""
        assert _apply_state_machine("LPS", next_has_transient=True) == "OLS"

    def test_invalid_frame_type(self) -> None:
        """Should raise ValueError for invalid frame type."""
        with pytest.raises(ValueError):
            _apply_state_machine("INVALID", next_has_transient=False)  # type: ignore

    def test_full_cycle(self) -> None:
        """Test complete transient cycle: OLS -> LSS -> ESH -> LPS -> OLS."""
        # Start steady
        state = "OLS"
        assert _apply_state_machine(state, False) == "OLS"

        # Transient detected -> transition to LSS
        state = _apply_state_machine("OLS", True)
        assert state == "LSS"

        # LSS forces ESH
        state = _apply_state_machine(state, False)
        assert state == "ESH"

        # No more transient -> exit to LPS
        state = _apply_state_machine(state, False)
        assert state == "LPS"

        # LPS forces OLS
        state = _apply_state_machine(state, False)
        assert state == "OLS"


class TestSSC:
    """Integration tests for SSC function."""

    def test_steady_state_sequence(self, stereo_sine_wave: np.ndarray) -> None:
        """Sequence of steady frames should be OLS."""
        # Steady sine wave -> no transient -> stay OLS
        frame_type = SSC(stereo_sine_wave, stereo_sine_wave, prev_frame_type="OLS")
        assert frame_type == "OLS"

    def test_steady_from_lps(self, stereo_sine_wave: np.ndarray) -> None:
        """LPS should force transition to OLS."""
        frame_type = SSC(stereo_sine_wave, stereo_sine_wave, prev_frame_type="LPS")
        assert frame_type == "OLS"

    def test_lss_forces_esh(self, stereo_sine_wave: np.ndarray) -> None:
        """LSS should force transition to ESH."""
        frame_type = SSC(stereo_sine_wave, stereo_sine_wave, prev_frame_type="LSS")
        assert frame_type == "ESH"

    def test_esh_to_lps_no_transient(self, stereo_sine_wave: np.ndarray) -> None:
        """ESH with no transient in next should go to LPS."""
        frame_type = SSC(stereo_sine_wave, stereo_sine_wave, prev_frame_type="ESH")
        assert frame_type == "LPS"

    def test_transient_triggers_lss(self) -> None:
        """Transient in next frame should trigger OLS -> LSS."""
        steady = np.zeros((2048, 2))
        steady[:, :] = 0.1  # Low constant signal

        # Create transient frame
        transient = np.zeros((2048, 2))
        transient[448 : 448 + 128 * 3, :] = 0.001
        transient[448 + 128 * 3 : 448 + 128 * 4, :] = 1.0

        frame_type = SSC(steady, transient, prev_frame_type="OLS")
        assert frame_type == "LSS"

    def test_transient_in_one_channel(self) -> None:
        """Transient in only one channel should still trigger LSS."""
        steady = np.ones((2048, 2)) * 0.1

        # Transient only in left channel
        transient = np.ones((2048, 2)) * 0.1
        transient[448 : 448 + 128 * 3, 0] = 0.001
        transient[448 + 128 * 3 : 448 + 128 * 4, 0] = 1.0

        frame_type = SSC(steady, transient, prev_frame_type="OLS")
        assert frame_type == "LSS"

    def test_multiple_transients_stay_esh(self) -> None:
        """Consecutive transients should keep ESH."""
        steady = np.ones((2048, 2)) * 0.1

        transient = np.zeros((2048, 2))
        transient[448 : 448 + 128 * 3, :] = 0.001
        transient[448 + 128 * 3 : 448 + 128 * 4, :] = 1.0

        frame_type = SSC(steady, transient, prev_frame_type="ESH")
        assert frame_type == "ESH"

    def test_silence_no_transient(self, stereo_silence: np.ndarray) -> None:
        """Silent frames should not detect transient."""
        frame_type = SSC(stereo_silence, stereo_silence, prev_frame_type="OLS")
        assert frame_type == "OLS"
