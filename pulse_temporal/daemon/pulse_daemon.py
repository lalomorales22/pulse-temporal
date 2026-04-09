"""PULSE background daemon.

Maintains continuous temporal state through a heartbeat loop,
tracking events, deadlines, circadian phase, and behavioral patterns.

v0.1: Basic daemon with event logging, deadline tracking, and
temporal context generation. The full heartbeat loop with external
event sources comes in v0.2.
"""

import threading
import time
from datetime import datetime
from typing import Optional, List

from ..encoder import PulseEncoder
from .state_db import StateDB


class PulseDaemon:
    """Background temporal processing daemon.

    Usage:
        daemon = PulseDaemon()
        daemon.add_deadline("project X", "2026-04-15T17:00:00")
        daemon.log_event(event_type="work_session_start")
        daemon.start()  # begins heartbeat loop in background

        context = daemon.get_temporal_context()
    """

    def __init__(
        self,
        db_path: str = "~/.pulse/state.db",
        heartbeat_interval: int = 60,
        model_name: Optional[str] = None,
    ):
        self.db = StateDB(db_path)
        self.encoder = PulseEncoder(model_name=model_name)
        self.heartbeat_interval = heartbeat_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat: Optional[datetime] = None

    def start(self):
        """Start the heartbeat loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _heartbeat_loop(self):
        """Main heartbeat: periodic state update."""
        while self._running:
            try:
                self._heartbeat()
            except Exception:
                pass  # daemon must not crash
            time.sleep(self.heartbeat_interval)

    def _heartbeat(self):
        """Single heartbeat tick: log the heartbeat event."""
        now = datetime.now()
        self.db.log_event(
            timestamp=now.isoformat(),
            event_type="heartbeat",
            source="daemon",
        )
        self._last_heartbeat = now

    def log_event(
        self,
        event_type: str = "generic",
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Log a temporal event."""
        return self.db.log_event(
            timestamp=timestamp,
            event_type=event_type,
            source="user",
            metadata=metadata,
        )

    def add_deadline(
        self,
        name: str,
        target_time: str,
        priority: str = "medium",
    ) -> int:
        """Register a deadline."""
        return self.db.add_deadline(name, target_time, priority)

    def complete_deadline(self, deadline_id: int):
        """Mark a deadline as completed."""
        self.db.complete_deadline(deadline_id)

    def get_temporal_context(self, t: Optional[str] = None) -> dict:
        """Get full temporal context package for LLM injection.

        This is the primary interface for the PULSE/MIND architecture.
        """
        if t is None:
            t = datetime.now()

        db_context = self.db.get_context_for_encoder()
        context = self.encoder.get_temporal_context(t, db_context)

        # Enrich with deadline details
        deadlines = self.db.get_active_deadlines()
        if deadlines:
            summaries = []
            now = datetime.now() if isinstance(t, str) else t
            if isinstance(now, str):
                now = datetime.fromisoformat(now)
            for d in deadlines:
                target = datetime.fromisoformat(d["target_time"])
                delta = target - now
                hours = delta.total_seconds() / 3600
                if hours > 0:
                    if hours > 48:
                        time_str = f"{hours / 24:.1f} days"
                    else:
                        time_str = f"{hours:.1f} hours"
                    summaries.append(f"{d['name']} in {time_str}")
                else:
                    summaries.append(f"{d['name']} OVERDUE by {-hours:.1f} hours")
            context["urgency_summary"] = "; ".join(summaries)
        else:
            context["urgency_summary"] = "no active deadlines"

        # Time since last interaction (non-heartbeat event)
        events = self.db.get_events(event_type=None, limit=10)
        user_events = [e for e in events if e.get("event_type") != "heartbeat"]
        if user_events:
            last = datetime.fromisoformat(user_events[0]["timestamp"])
            now = datetime.now()
            delta_h = (now - last).total_seconds() / 3600
            if delta_h < 1:
                context["time_since_last_interaction"] = f"{delta_h * 60:.0f} minutes"
            elif delta_h < 24:
                context["time_since_last_interaction"] = f"{delta_h:.1f} hours"
            else:
                context["time_since_last_interaction"] = f"{delta_h / 24:.1f} days"
        else:
            context["time_since_last_interaction"] = "no prior interactions"

        return context

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"PulseDaemon(status={status}, interval={self.heartbeat_interval}s)"
