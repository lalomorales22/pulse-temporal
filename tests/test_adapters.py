"""Tests for the PULSE event source adapters."""

import os
import tempfile
import pytest
from datetime import datetime, timedelta

from pulse_temporal.adapters.git_adapter import GitAdapter
from pulse_temporal.adapters.ical_adapter import ICalAdapter
from pulse_temporal.daemon import PulseDaemon


class TestGitAdapter:
    """Test the git event source adapter."""

    def test_init_with_valid_repo(self):
        """Initialize with the current repo (we're in a git repo)."""
        adapter = GitAdapter(".")
        assert adapter.repo_path.exists()

    def test_init_with_invalid_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Not a git repository"):
                GitAdapter(tmpdir)

    def test_get_commits(self):
        adapter = GitAdapter(".")
        commits = adapter.get_commits(limit=5)
        assert isinstance(commits, list)
        # This repo has commits
        assert len(commits) > 0
        assert "hash" in commits[0]
        assert "timestamp" in commits[0]
        assert "message" in commits[0]
        assert "author" in commits[0]

    def test_get_commits_with_limit(self):
        adapter = GitAdapter(".")
        commits = adapter.get_commits(limit=2)
        assert len(commits) <= 2

    def test_get_branch_info(self):
        adapter = GitAdapter(".")
        info = adapter.get_branch_info()
        assert "branch" in info
        assert "commits_ahead" in info
        assert "commits_behind" in info
        assert isinstance(info["branch"], str)

    def test_get_activity_summary(self):
        adapter = GitAdapter(".")
        summary = adapter.get_activity_summary(hours=720)  # 30 days
        assert "total_commits" in summary
        assert "activity_level" in summary
        assert summary["activity_level"] in ("none", "light", "moderate", "active", "intense")

    def test_sync_to_daemon(self):
        adapter = GitAdapter(".")
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = PulseDaemon(db_path=os.path.join(tmpdir, "test.db"))
            logged = adapter.sync(daemon, limit=3)
            assert logged <= 3

    def test_get_file_churn(self):
        adapter = GitAdapter(".")
        churn = adapter.get_file_churn(hours=720, limit=5)
        assert isinstance(churn, list)
        if churn:
            assert "file" in churn[0]
            assert "changes" in churn[0]

    def test_repr(self):
        adapter = GitAdapter(".")
        assert "GitAdapter" in repr(adapter)


class TestICalAdapter:
    """Test the iCal event source adapter."""

    SAMPLE_ICS = """\
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Test//EN
BEGIN:VEVENT
DTSTART:{start}
DTEND:{end}
SUMMARY:Team standup
LOCATION:Zoom
STATUS:CONFIRMED
END:VEVENT
BEGIN:VEVENT
DTSTART:{start2}
DTEND:{end2}
SUMMARY:Code review
END:VEVENT
END:VCALENDAR
"""

    def _make_ics(self, tmpdir):
        """Create a sample .ics file with events around now."""
        now = datetime.now()
        start = now + timedelta(hours=1)
        end = start + timedelta(minutes=30)
        start2 = now + timedelta(hours=3)
        end2 = start2 + timedelta(hours=1)

        content = self.SAMPLE_ICS.format(
            start=start.strftime("%Y%m%dT%H%M%S"),
            end=end.strftime("%Y%m%dT%H%M%S"),
            start2=start2.strftime("%Y%m%dT%H%M%S"),
            end2=end2.strftime("%Y%m%dT%H%M%S"),
        )
        path = os.path.join(tmpdir, "test.ics")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_init_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_ics(tmpdir)
            adapter = ICalAdapter(path)
            assert adapter.source == path

    def test_init_with_url(self):
        adapter = ICalAdapter("https://example.com/cal.ics")
        assert "https" in adapter.source

    def test_file_not_found(self):
        adapter = ICalAdapter("/nonexistent/path.ics")
        with pytest.raises(FileNotFoundError):
            adapter.get_events()

    def test_get_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_ics(tmpdir)
            adapter = ICalAdapter(path)
            events = adapter.get_events()
            assert len(events) == 2
            assert events[0]["summary"] == "Team standup"
            assert events[1]["summary"] == "Code review"

    def test_get_events_time_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_ics(tmpdir)
            adapter = ICalAdapter(path)
            now = datetime.now()
            # Only events in the next 2 hours
            events = adapter.get_events(
                after=now,
                before=now + timedelta(hours=2),
            )
            assert len(events) == 1
            assert events[0]["summary"] == "Team standup"

    def test_get_today_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_ics(tmpdir)
            adapter = ICalAdapter(path)
            summary = adapter.get_today_summary()
            assert "total_events" in summary
            assert "busyness" in summary
            assert "meeting_minutes" in summary
            assert summary["total_events"] >= 0

    def test_sync_to_daemon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_ics(tmpdir)
            adapter = ICalAdapter(path)
            daemon = PulseDaemon(db_path=os.path.join(tmpdir, "test.db"))
            logged = adapter.sync(daemon)
            assert logged == 2

    def test_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_ics(tmpdir)
            adapter = ICalAdapter(path)
            adapter.get_events()  # populates cache
            assert adapter._raw is not None
            adapter.invalidate()
            assert adapter._raw is None

    def test_repr(self):
        adapter = ICalAdapter("test.ics")
        assert "ICalAdapter" in repr(adapter)
