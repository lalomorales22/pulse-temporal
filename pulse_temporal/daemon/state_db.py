"""SQLite state management for the PULSE daemon.

Stores events, deadlines, behavioral patterns, and temporal priors.
Uses a single SQLite file for zero-dependency persistence.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


class StateDB:
    """Lightweight SQLite store for temporal state."""

    def __init__(self, db_path: str = "~/.pulse/state.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT DEFAULT 'generic',
                source TEXT DEFAULT 'manual',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS deadlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                target_time TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                completed INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                data TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_deadlines_target ON deadlines(target_time);
        """)
        self._conn.commit()

    def log_event(
        self,
        timestamp: Optional[str] = None,
        event_type: str = "generic",
        source: str = "manual",
        metadata: Optional[dict] = None,
    ) -> int:
        """Log a temporal event. Returns the event ID."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})
        cur = self._conn.execute(
            "INSERT INTO events (timestamp, event_type, source, metadata) VALUES (?, ?, ?, ?)",
            (timestamp, event_type, source, meta_json),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_events(
        self,
        since: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Retrieve recent events."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_event_history(self, limit: int = 100) -> List[str]:
        """Get recent event timestamps as a list (for PulseEncoder context)."""
        rows = self._conn.execute(
            "SELECT timestamp FROM events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["timestamp"] for r in rows]

    def add_deadline(
        self,
        name: str,
        target_time: str,
        priority: str = "medium",
        metadata: Optional[dict] = None,
    ) -> int:
        """Register a deadline. Returns the deadline ID."""
        meta_json = json.dumps(metadata or {})
        cur = self._conn.execute(
            "INSERT INTO deadlines (name, target_time, priority, metadata) VALUES (?, ?, ?, ?)",
            (name, target_time, priority, meta_json),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_active_deadlines(self) -> List[Dict]:
        """Get all non-completed deadlines."""
        rows = self._conn.execute(
            "SELECT * FROM deadlines WHERE completed = 0 ORDER BY target_time ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_deadline(self, deadline_id: int):
        """Mark a deadline as completed."""
        self._conn.execute(
            "UPDATE deadlines SET completed = 1 WHERE id = ?", (deadline_id,)
        )
        self._conn.commit()

    def get_context_for_encoder(self) -> dict:
        """Build a context dict suitable for PulseEncoder.encode()."""
        events = self.get_event_history(limit=200)
        deadlines = self.get_active_deadlines()

        deadline_times = [d["target_time"] for d in deadlines]

        context = {
            "event_history": events,
        }
        if deadline_times:
            context["deadline"] = deadline_times[0]  # most urgent
            context["deadlines"] = deadline_times

        return context

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
