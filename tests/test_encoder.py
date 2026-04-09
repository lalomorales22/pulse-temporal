"""Tests for PulseEncoder -- the core embedding model."""

import numpy as np
import pytest
from datetime import datetime, timedelta

from pulse_temporal import PulseEncoder


@pytest.fixture
def pulse():
    return PulseEncoder()


class TestBasicEncoding:
    def test_encode_returns_correct_shape(self, pulse):
        emb = pulse.encode("2026-04-09T14:30:00")
        assert emb.shape == (128,)
        assert emb.dtype == np.float32

    def test_encode_is_normalized(self, pulse):
        emb = pulse.encode("2026-04-09T14:30:00")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_encode_datetime_object(self, pulse):
        dt = datetime(2026, 4, 9, 14, 30, 0)
        emb = pulse.encode(dt)
        assert emb.shape == (128,)

    def test_encode_unix_timestamp(self, pulse):
        ts = datetime(2026, 4, 9, 14, 30, 0).timestamp()
        emb = pulse.encode(ts)
        assert emb.shape == (128,)

    def test_same_input_same_output(self, pulse):
        emb1 = pulse.encode("2026-04-09T14:30:00")
        emb2 = pulse.encode("2026-04-09T14:30:00")
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_times_different_embeddings(self, pulse):
        emb1 = pulse.encode("2026-04-09T09:00:00")
        emb2 = pulse.encode("2026-04-09T21:00:00")
        assert not np.allclose(emb1, emb2)


class TestContextualEncoding:
    def test_deadline_changes_embedding(self, pulse):
        base = pulse.encode("2026-04-09T14:00:00")
        with_deadline = pulse.encode("2026-04-09T14:00:00", context={
            "deadline": "2026-04-09T15:00:00"
        })
        # Same timestamp, different context -> different embedding
        assert not np.allclose(base, with_deadline)

    def test_urgency_increases_near_deadline(self, pulse):
        far = pulse.encode("2026-04-09T09:00:00", context={
            "deadline": "2026-04-09T17:00:00"  # 8 hours away
        })
        near = pulse.encode("2026-04-09T16:00:00", context={
            "deadline": "2026-04-09T17:00:00"  # 1 hour away
        })
        # These should be quite different
        sim = pulse.similarity(far, near)
        assert sim < 0.9  # not too similar

    def test_event_history_changes_embedding(self, pulse):
        base = pulse.encode("2026-04-09T14:00:00")
        with_history = pulse.encode("2026-04-09T14:00:00", context={
            "event_history": [
                "2026-04-09T13:00:00",
                "2026-04-09T12:00:00",
                "2026-04-09T11:00:00",
                "2026-04-09T10:00:00",
            ]
        })
        assert not np.allclose(base, with_history)


class TestExperientialSimilarity:
    """The core PULSE insight: experientially similar moments cluster together."""

    def test_crunch_times_cluster(self, pulse):
        """Two pre-deadline crunches should be more similar to each other
        than either is to a relaxed weekend."""
        monday_crunch = pulse.encode("2026-04-13T14:00:00", context={
            "deadline": "2026-04-13T17:00:00",
            "events_today": 6,
            "sleep_hours": 5,
        })
        wednesday_crunch = pulse.encode("2026-04-15T10:00:00", context={
            "deadline": "2026-04-15T12:00:00",
            "events_today": 4,
            "sleep_hours": 6,
        })
        saturday_chill = pulse.encode("2026-04-11T14:00:00", context={
            "deadline": None,
            "events_today": 0,
            "sleep_hours": 9,
        })

        crunch_sim = pulse.similarity(monday_crunch, wednesday_crunch)
        crunch_vs_chill = pulse.similarity(monday_crunch, saturday_chill)

        # Crunch times should be more similar to each other
        assert crunch_sim > crunch_vs_chill

    def test_same_circadian_phase_clusters(self, pulse):
        """Same time of day on different dates should be somewhat similar."""
        monday_9am = pulse.encode("2026-04-13T09:00:00")
        tuesday_9am = pulse.encode("2026-04-14T09:00:00")
        monday_9pm = pulse.encode("2026-04-13T21:00:00")

        same_time_sim = pulse.similarity(monday_9am, tuesday_9am)
        diff_time_sim = pulse.similarity(monday_9am, monday_9pm)

        assert same_time_sim > diff_time_sim


class TestBatchEncoding:
    def test_batch_shape(self, pulse):
        timestamps = [
            "2026-04-09T09:00:00",
            "2026-04-09T12:00:00",
            "2026-04-09T15:00:00",
        ]
        batch = pulse.encode_batch(timestamps)
        assert batch.shape == (3, 128)

    def test_batch_matches_individual(self, pulse):
        timestamps = ["2026-04-09T09:00:00", "2026-04-09T15:00:00"]
        batch = pulse.encode_batch(timestamps)
        individual = [pulse.encode(t) for t in timestamps]
        for i, emb in enumerate(individual):
            np.testing.assert_array_almost_equal(batch[i], emb)


class TestSimilarityMatrix:
    def test_matrix_shape(self, pulse):
        embeddings = [pulse.encode(f"2026-04-0{i}T12:00:00") for i in range(1, 5)]
        mat = pulse.similarity_matrix(embeddings)
        assert mat.shape == (4, 4)

    def test_diagonal_is_one(self, pulse):
        embeddings = [pulse.encode(f"2026-04-0{i}T12:00:00") for i in range(1, 4)]
        mat = pulse.similarity_matrix(embeddings)
        np.testing.assert_array_almost_equal(np.diag(mat), np.ones(3), decimal=5)


class TestDecompose:
    def test_decompose_returns_all_layers(self, pulse):
        result = pulse.decompose("2026-04-09T14:00:00")
        expected_keys = {
            "log_time", "oscillators", "circadian", "calendar",
            "urgency", "temporal_state", "prediction_error",
        }
        assert set(result.keys()) == expected_keys

    def test_decompose_dimensions(self, pulse):
        result = pulse.decompose("2026-04-09T14:00:00")
        assert result["log_time"].shape == (8,)
        assert result["oscillators"].shape == (32,)
        assert result["circadian"].shape == (8,)
        assert result["calendar"].shape == (24,)
        assert result["urgency"].shape == (8,)
        assert result["temporal_state"].shape == (32,)
        assert result["prediction_error"].shape == (16,)


class TestTemporalContext:
    def test_context_package_keys(self, pulse):
        ctx = pulse.get_temporal_context("2026-04-09T14:00:00")
        assert "embedding" in ctx
        assert "circadian_phase" in ctx
        assert "cognitive_capacity" in ctx
        assert "energy_level" in ctx
        assert "urgency_level" in ctx

    def test_context_afternoon_peak(self, pulse):
        ctx = pulse.get_temporal_context("2026-04-09T15:00:00")
        assert ctx["circadian_phase"] == "afternoon_peak"

    def test_context_deep_night(self, pulse):
        ctx = pulse.get_temporal_context("2026-04-09T03:00:00")
        assert ctx["circadian_phase"] == "deep_night"
