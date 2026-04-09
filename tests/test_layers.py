"""Tests for individual PULSE layers."""

import numpy as np
import pytest
from datetime import datetime

from pulse_temporal.layers import (
    LogTimeLayer,
    OscillatorLayer,
    CircadianLayer,
    CalendarLayer,
    UrgencyLayer,
    TemporalStateLayer,
    PredictionErrorLayer,
)


class TestLogTimeLayer:
    def test_output_shape(self):
        layer = LogTimeLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 30))
        assert result.shape == (8,)

    def test_values_bounded(self):
        layer = LogTimeLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 30))
        assert np.all(result >= 0) and np.all(result <= 2.0)

    def test_different_times_different_outputs(self):
        layer = LogTimeLayer()
        a = layer.encode(datetime(2026, 4, 9, 9, 0))
        b = layer.encode(datetime(2026, 4, 9, 21, 0))
        assert not np.allclose(a, b)


class TestOscillatorLayer:
    def test_output_shape(self):
        layer = OscillatorLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 30))
        assert result.shape == (32,)

    def test_values_in_range(self):
        layer = OscillatorLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 30))
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_periodicity(self):
        """Same time, exactly 24 hours apart: the 24h oscillator should match."""
        layer = OscillatorLayer()
        t1 = layer.encode(datetime(2026, 4, 9, 12, 0))
        t2 = layer.encode(datetime(2026, 4, 10, 12, 0))
        # The 24h frequency component (index 12-13) should be very similar
        assert abs(t1[12] - t2[12]) < 0.01
        assert abs(t1[13] - t2[13]) < 0.01


class TestCircadianLayer:
    def test_output_shape(self):
        layer = CircadianLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 30))
        assert result.shape == (8,)

    def test_peak_cognition_midmorning(self):
        layer = CircadianLayer()
        morning = layer.encode(datetime(2026, 4, 9, 10, 30))
        night = layer.encode(datetime(2026, 4, 9, 3, 0))
        # Cognitive capacity (index 6) should be higher in morning
        assert morning[6] > night[6]

    def test_post_lunch_dip(self):
        layer = CircadianLayer()
        peak = layer.encode(datetime(2026, 4, 9, 10, 30))
        dip = layer.encode(datetime(2026, 4, 9, 14, 0))
        # Post-lunch cognitive dip
        assert peak[6] > dip[6]


class TestCalendarLayer:
    def test_output_shape(self):
        layer = CalendarLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 30))
        assert result.shape == (24,)

    def test_weekend_detection(self):
        layer = CalendarLayer()
        # April 11, 2026 is Saturday
        saturday = layer.encode(datetime(2026, 4, 11, 12, 0))
        # April 13, 2026 is Monday
        monday = layer.encode(datetime(2026, 4, 13, 12, 0))
        # is_weekend is at index 10
        assert saturday[10] == 1.0
        assert monday[10] == 0.0

    def test_business_hours(self):
        layer = CalendarLayer()
        # Monday 2pm = business hours
        biz = layer.encode(datetime(2026, 4, 13, 14, 0))
        # Monday 11pm = not business hours
        late = layer.encode(datetime(2026, 4, 13, 23, 0))
        # is_business is at index 21
        assert biz[21] == 1.0
        assert late[21] == 0.0


class TestUrgencyLayer:
    def test_no_deadline_is_zero(self):
        layer = UrgencyLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 0))
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_urgency_increases_near_deadline(self):
        layer = UrgencyLayer()
        far = layer.encode(
            datetime(2026, 4, 9, 9, 0),
            context={"deadline": "2026-04-09T17:00:00"}  # 8 hours
        )
        near = layer.encode(
            datetime(2026, 4, 9, 16, 0),
            context={"deadline": "2026-04-09T17:00:00"}  # 1 hour
        )
        # Medium urgency (index 1) should be higher when near
        assert near[1] > far[1]

    def test_overdue_detection(self):
        layer = UrgencyLayer()
        overdue = layer.encode(
            datetime(2026, 4, 9, 18, 0),
            context={"deadline": "2026-04-09T17:00:00"}  # 1 hour past
        )
        # is_overdue (index 4) should be high
        assert overdue[4] > 0.5

    def test_multiple_deadlines(self):
        layer = UrgencyLayer()
        result = layer.encode(
            datetime(2026, 4, 9, 14, 0),
            context={"deadlines": ["2026-04-09T17:00:00", "2026-04-10T09:00:00"]}
        )
        # n_deadlines (index 6) should reflect 2 deadlines
        assert result[6] > 0


class TestTemporalStateLayer:
    def test_output_shape(self):
        layer = TemporalStateLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 0))
        assert result.shape == (32,)

    def test_context_features(self):
        layer = TemporalStateLayer()
        result = layer.encode(
            datetime(2026, 4, 9, 14, 0),
            context={"events_today": 8, "sleep_hours": 5}
        )
        # events_today normalized (index 0) should be 8/12
        assert abs(result[0] - 8 / 12) < 0.01

    def test_event_history_activates_features(self):
        layer = TemporalStateLayer()
        no_history = layer.encode(datetime(2026, 4, 9, 14, 0))
        with_history = layer.encode(
            datetime(2026, 4, 9, 14, 0),
            context={
                "event_history": [
                    "2026-04-09T13:30:00",
                    "2026-04-09T13:00:00",
                    "2026-04-09T12:00:00",
                ]
            }
        )
        # Decay features (indices 4-11) should differ
        assert not np.allclose(no_history[4:12], with_history[4:12])


class TestPredictionErrorLayer:
    def test_output_shape(self):
        layer = PredictionErrorLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 0))
        assert result.shape == (16,)

    def test_no_context_is_zero(self):
        layer = PredictionErrorLayer()
        result = layer.encode(datetime(2026, 4, 9, 14, 0))
        np.testing.assert_array_equal(result, np.zeros(16))

    def test_explicit_prediction_error(self):
        layer = PredictionErrorLayer()
        result = layer.encode(
            datetime(2026, 4, 9, 14, 0),
            context={"t_expected": "2026-04-09T12:00:00"}  # 2 hours late
        )
        # Error features should be non-zero
        assert np.any(result[:8] != 0)
        # Direction (index 5) should be positive (late)
        assert result[5] > 0

    def test_early_vs_late(self):
        layer = PredictionErrorLayer()
        early = layer.encode(
            datetime(2026, 4, 9, 10, 0),
            context={"t_expected": "2026-04-09T12:00:00"}
        )
        late = layer.encode(
            datetime(2026, 4, 9, 14, 0),
            context={"t_expected": "2026-04-09T12:00:00"}
        )
        # Direction should differ
        assert early[5] < 0  # early
        assert late[5] > 0   # late
