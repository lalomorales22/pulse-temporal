"""Continuous-time event history state encoding.

Inspired by the Neural Hawkes Process continuous-time LSTM:
c(t) = c_bar + (c - c_bar) * exp(-delta * (t - t_last))

Memory cells decay exponentially between events, giving PULSE a
running sense of "what's been happening" that evolves even between
observations. For v0.1, uses fixed multi-scale exponential kernels
rather than learned parameters.
"""

import numpy as np
from datetime import datetime
from typing import Optional

_EPS = 1e-8

# Exponential decay time constants (in hours) spanning multiple scales
_DECAY_SCALES_HOURS = np.array([
    0.25,    # 15 minutes -- captures very recent activity bursts
    1.0,     # 1 hour -- short-term context
    4.0,     # 4 hours -- session-level
    12.0,    # 12 hours -- half-day
    24.0,    # 1 day -- daily patterns
    72.0,    # 3 days -- multi-day context
    168.0,   # 1 week -- weekly patterns
    720.0,   # 30 days -- monthly patterns
], dtype=np.float64)


def _parse_event_time(event, reference_dt: datetime) -> Optional[float]:
    """Extract hours-before-reference from an event entry."""
    if isinstance(event, datetime):
        delta = (reference_dt - event).total_seconds() / 3600.0
        return delta if delta >= 0 else None
    if isinstance(event, str):
        try:
            evt_dt = datetime.fromisoformat(event)
            delta = (reference_dt - evt_dt).total_seconds() / 3600.0
            return delta if delta >= 0 else None
        except ValueError:
            return None
    if isinstance(event, (tuple, list)) and len(event) >= 1:
        return _parse_event_time(event[0], reference_dt)
    if isinstance(event, dict) and "time" in event:
        return _parse_event_time(event["time"], reference_dt)
    return None


class TemporalStateLayer:
    """Produces 32D vector encoding temporal state from event history and context."""

    dim = 32

    def encode(self, dt: datetime, context: Optional[dict] = None) -> np.ndarray:
        context = context or {}
        features = np.zeros(self.dim, dtype=np.float32)

        # --- Context-derived features (4D) [indices 0-3] ---
        events_today = context.get("events_today", 0)
        sleep_hours = context.get("sleep_hours", 7.0)
        hours_active = context.get("hours_active")

        if hours_active is None:
            # Estimate from time of day and sleep
            hour = dt.hour + dt.minute / 60.0
            wake_hour = max(0, hour - (8 - min(sleep_hours, 8)))
            hours_active = max(0, min(wake_hour, 18))

        features[0] = min(events_today / 12.0, 1.0)  # normalized event load
        features[1] = min(sleep_hours / 10.0, 1.0)    # normalized sleep
        features[2] = min(hours_active / 16.0, 1.0)   # normalized active time
        # Stress estimate: high events + low sleep = high stress
        features[3] = min(1.0, (events_today / 8.0) * (1.0 - sleep_hours / 10.0) + 0.1)

        # --- Event history features (28D) [indices 4-31] ---
        event_history = context.get("event_history", [])

        if not event_history:
            # No history: return context features + baseline state
            # Set decay features to 0 (no events), stats to defaults
            features[28] = 0.5  # neutral fatigue estimate
            features[29] = 0.5  # neutral engagement
            features[30] = 1.0  # idle state (no events)
            return features

        # Parse event times to hours-before-now
        hours_ago = []
        for event in event_history:
            h = _parse_event_time(event, dt)
            if h is not None:
                hours_ago.append(h)

        if not hours_ago:
            features[28] = 0.5
            features[29] = 0.5
            features[30] = 1.0
            return features

        hours_ago = np.array(sorted(hours_ago), dtype=np.float64)

        # Multi-scale exponential decay sums (8D) [indices 4-11]
        # Each dimension captures event density at a different time scale
        for i, tau in enumerate(_DECAY_SCALES_HOURS):
            decay_weights = np.exp(-hours_ago / tau)
            features[4 + i] = float(np.sum(decay_weights)) / (len(hours_ago) + 1)

        # Inter-event interval statistics (4D) [indices 12-15]
        if len(hours_ago) >= 2:
            intervals = np.diff(hours_ago)
            intervals = intervals[intervals > _EPS]
            if len(intervals) > 0:
                features[12] = float(np.mean(intervals)) / 24.0  # mean interval (normalized to days)
                features[13] = float(np.std(intervals)) / 24.0   # interval variability
                # Burstiness: (std - mean) / (std + mean), ranges [-1, 1]
                mu, sigma = float(np.mean(intervals)), float(np.std(intervals))
                features[14] = (sigma - mu) / (sigma + mu + _EPS)
                # Trend: are intervals getting shorter or longer?
                if len(intervals) >= 3:
                    half = len(intervals) // 2
                    recent_mean = float(np.mean(intervals[:half]))
                    older_mean = float(np.mean(intervals[half:]))
                    features[15] = np.tanh((recent_mean - older_mean) / (older_mean + _EPS))

        # Event rate at multiple scales (4D) [indices 16-19]
        windows_hours = [1.0, 6.0, 24.0, 168.0]
        for i, window in enumerate(windows_hours):
            count = float(np.sum(hours_ago <= window))
            # Normalize by window size to get events-per-hour
            features[16 + i] = min(count / window, 1.0)

        # Recency features (4D) [indices 20-23]
        most_recent = float(hours_ago[0]) if len(hours_ago) > 0 else 999.0
        features[20] = np.exp(-most_recent)                    # very short-term recency
        features[21] = np.exp(-most_recent / 4.0)              # session recency
        features[22] = np.exp(-most_recent / 24.0)             # daily recency
        features[23] = float(np.log1p(most_recent)) / 10.0     # log time since last event

        # Session detection (4D) [indices 24-27]
        # A "session" = cluster of events with < 30 min gaps
        session_gap = 0.5  # hours
        if len(hours_ago) >= 2:
            intervals_from_now = hours_ago
            in_session = intervals_from_now < session_gap
            features[24] = float(np.sum(in_session)) / max(len(hours_ago), 1)  # session density
            # Current session length (hours since first event in current session)
            session_events = hours_ago[hours_ago < 4.0]  # look at last 4 hours
            if len(session_events) >= 2:
                session_intervals = np.diff(session_events)
                session_mask = session_intervals < session_gap
                # Find the session boundary
                if np.any(~session_mask):
                    boundary_idx = np.where(~session_mask)[0][0]
                    session_len = float(session_events[boundary_idx])
                else:
                    session_len = float(session_events[-1])
                features[25] = min(session_len / 4.0, 1.0)  # session duration normalized
            features[26] = 1.0 if most_recent < session_gap else 0.0  # currently in session
        features[27] = float(len(hours_ago)) / 100.0  # total event count (normalized)

        # Derived state estimates (4D) [indices 28-31]
        # Fatigue: increases with hours active, decreases with sleep
        fatigue = min(1.0, hours_active / 14.0) * (1.0 - 0.3 * min(sleep_hours / 8.0, 1.0))
        features[28] = fatigue
        # Engagement: high recent activity + in session = engaged
        features[29] = 0.7 * features[26] + 0.3 * features[20]
        # Activity state: [active, idle, resting]
        if most_recent < 0.1:
            features[30] = 1.0   # active
        elif most_recent < 2.0:
            features[31] = 0.5   # idle
        else:
            features[30] = 0.0   # resting (default)

        return features
