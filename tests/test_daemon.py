"""Tests for the PULSE daemon and state DB."""

import os
import tempfile
import pytest
from datetime import datetime

from pulse_temporal.daemon import PulseDaemon, StateDB


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = StateDB(db_path=path)
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def tmp_daemon():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    daemon = PulseDaemon(db_path=path, heartbeat_interval=999)
    yield daemon
    daemon.stop()
    os.unlink(path)


class TestStateDB:
    def test_log_and_retrieve_event(self, tmp_db):
        tmp_db.log_event(timestamp="2026-04-09T14:00:00", event_type="test")
        events = tmp_db.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "test"

    def test_multiple_events(self, tmp_db):
        for i in range(5):
            tmp_db.log_event(timestamp=f"2026-04-09T{10+i}:00:00", event_type="work")
        events = tmp_db.get_events()
        assert len(events) == 5

    def test_event_history_returns_timestamps(self, tmp_db):
        tmp_db.log_event(timestamp="2026-04-09T14:00:00")
        tmp_db.log_event(timestamp="2026-04-09T15:00:00")
        history = tmp_db.get_event_history()
        assert len(history) == 2
        assert all(isinstance(t, str) for t in history)

    def test_add_and_get_deadlines(self, tmp_db):
        tmp_db.add_deadline("project X", "2026-04-15T17:00:00")
        deadlines = tmp_db.get_active_deadlines()
        assert len(deadlines) == 1
        assert deadlines[0]["name"] == "project X"

    def test_complete_deadline(self, tmp_db):
        did = tmp_db.add_deadline("project X", "2026-04-15T17:00:00")
        tmp_db.complete_deadline(did)
        assert len(tmp_db.get_active_deadlines()) == 0

    def test_context_for_encoder(self, tmp_db):
        tmp_db.log_event(timestamp="2026-04-09T14:00:00")
        tmp_db.add_deadline("test", "2026-04-10T12:00:00")
        ctx = tmp_db.get_context_for_encoder()
        assert "event_history" in ctx
        assert "deadline" in ctx


class TestPulseDaemon:
    def test_log_event(self, tmp_daemon):
        eid = tmp_daemon.log_event(event_type="test_event")
        assert eid > 0

    def test_add_deadline(self, tmp_daemon):
        did = tmp_daemon.add_deadline("test deadline", "2026-04-15T17:00:00")
        assert did > 0

    def test_get_temporal_context(self, tmp_daemon):
        tmp_daemon.log_event(event_type="start", timestamp="2026-04-09T13:00:00")
        tmp_daemon.add_deadline("project", "2026-04-10T17:00:00")
        ctx = tmp_daemon.get_temporal_context("2026-04-09T14:00:00")
        assert "embedding" in ctx
        assert "circadian_phase" in ctx
        assert "urgency_summary" in ctx

    def test_start_stop(self, tmp_daemon):
        tmp_daemon.start()
        assert tmp_daemon.is_running
        tmp_daemon.stop()
        assert not tmp_daemon.is_running
