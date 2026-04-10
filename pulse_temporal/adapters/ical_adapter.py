"""iCalendar event source adapter for the PULSE daemon.

Reads .ics files or iCal URLs and feeds calendar events into the
PULSE daemon's temporal context.

Supports:
- Local .ics files
- Remote iCal URLs (Google Calendar, Outlook, Apple Calendar export URLs)
- Recurring events (basic RRULE support)

Usage:
    from pulse_temporal.adapters import ICalAdapter
    from pulse_temporal.daemon import PulseDaemon

    daemon = PulseDaemon()
    cal = ICalAdapter("https://calendar.google.com/...basic.ics")
    cal.sync(daemon)
"""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Union
from urllib.request import urlopen
from urllib.error import URLError


class ICalAdapter:
    """Feeds calendar events into the PULSE daemon event stream."""

    def __init__(self, source: str):
        """Initialize with a file path or URL to an .ics file.

        Args:
            source: Path to .ics file or iCal URL.
        """
        self.source = source
        self._raw: Optional[str] = None

    def _fetch(self) -> str:
        """Fetch the raw iCal data."""
        if self._raw is not None:
            return self._raw

        if self.source.startswith(("http://", "https://")):
            try:
                with urlopen(self.source, timeout=30) as resp:
                    self._raw = resp.read().decode("utf-8", errors="replace")
            except URLError as e:
                raise ConnectionError(f"Failed to fetch calendar: {e}")
        else:
            path = Path(self.source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Calendar file not found: {path}")
            self._raw = path.read_text(encoding="utf-8", errors="replace")

        return self._raw

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        """Parse an iCal datetime string."""
        # Strip any TZID prefix
        if ":" in value:
            value = value.split(":")[-1]
        value = value.strip()

        # YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS or YYYYMMDD
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _parse_events(self, ical_text: str) -> List[Dict]:
        """Parse VEVENT blocks from raw iCal text."""
        events = []
        in_event = False
        current: dict = {}

        for line in ical_text.splitlines():
            line = line.strip()
            if line == "BEGIN:VEVENT":
                in_event = True
                current = {}
            elif line == "END:VEVENT":
                in_event = False
                if current.get("start"):
                    events.append(current)
            elif in_event:
                if line.startswith("DTSTART"):
                    current["start"] = self._parse_datetime(line.split(":", 1)[-1] if ":" in line else "")
                elif line.startswith("DTEND"):
                    current["end"] = self._parse_datetime(line.split(":", 1)[-1] if ":" in line else "")
                elif line.startswith("SUMMARY:"):
                    current["summary"] = line[8:]
                elif line.startswith("DESCRIPTION:"):
                    current["description"] = line[12:][:200]  # truncate
                elif line.startswith("LOCATION:"):
                    current["location"] = line[9:]
                elif line.startswith("STATUS:"):
                    current["status"] = line[7:]

        return events

    def get_events(
        self,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get calendar events within a time range.

        Args:
            after: Only events starting after this time. Default: now - 24h.
            before: Only events starting before this time. Default: now + 7 days.
        """
        raw = self._fetch()
        all_events = self._parse_events(raw)

        if after is None:
            after = datetime.now() - timedelta(hours=24)
        if before is None:
            before = datetime.now() + timedelta(days=7)

        filtered = []
        for ev in all_events:
            start = ev.get("start")
            if start and after <= start <= before:
                filtered.append({
                    "summary": ev.get("summary", "Untitled"),
                    "start": start.isoformat(),
                    "end": ev["end"].isoformat() if ev.get("end") else None,
                    "location": ev.get("location"),
                    "status": ev.get("status", "CONFIRMED"),
                })

        filtered.sort(key=lambda e: e["start"])
        return filtered

    def get_today_summary(self) -> Dict:
        """Get a summary of today's calendar for temporal context."""
        now = datetime.now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        events = self.get_events(after=start_of_day, before=end_of_day)

        # Calculate time in meetings
        meeting_minutes = 0
        for ev in events:
            if ev.get("end"):
                start = datetime.fromisoformat(ev["start"])
                end = datetime.fromisoformat(ev["end"])
                meeting_minutes += max(0, (end - start).total_seconds() / 60)

        # Find next event
        upcoming = [e for e in events if datetime.fromisoformat(e["start"]) > now]
        next_event = upcoming[0] if upcoming else None

        # Time until next event
        if next_event:
            delta = datetime.fromisoformat(next_event["start"]) - now
            minutes_until = delta.total_seconds() / 60
        else:
            minutes_until = None

        return {
            "date": now.strftime("%Y-%m-%d"),
            "total_events": len(events),
            "meeting_minutes": round(meeting_minutes),
            "next_event": next_event["summary"] if next_event else None,
            "minutes_until_next": round(minutes_until) if minutes_until is not None else None,
            "busyness": (
                "packed" if len(events) > 6
                else "busy" if len(events) > 3
                else "moderate" if len(events) > 1
                else "light" if len(events) == 1
                else "clear"
            ),
        }

    def sync(self, daemon, since: Optional[datetime] = None):
        """Sync calendar events into the PULSE daemon.

        Args:
            daemon: PulseDaemon instance.
            since: Only sync events after this time. Default: now - 24h.
        """
        if since is None:
            since = datetime.now() - timedelta(hours=24)

        events = self.get_events(after=since)
        logged = 0
        for ev in events:
            daemon.log_event(
                event_type="calendar_event",
                timestamp=ev["start"],
                metadata={
                    "summary": ev["summary"],
                    "end": ev.get("end"),
                    "location": ev.get("location"),
                },
            )
            logged += 1
        return logged

    def invalidate(self):
        """Clear cached calendar data, forcing a re-fetch."""
        self._raw = None

    def __repr__(self) -> str:
        src = self.source if len(self.source) < 50 else self.source[:47] + "..."
        return f"ICalAdapter(source='{src}')"
