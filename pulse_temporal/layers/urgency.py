"""Hyperbolic deadline proximity encoding.

From behavioral economics: humans perceive urgency via hyperbolic
discounting V_d = V / (1 + k*d). The perceived urgency at T-1 hour
is far more than double the urgency at T-2 hours.

This layer encodes deadline proximity at multiple discount rates,
capturing everything from long-range awareness to last-minute panic.
"""

import numpy as np
from datetime import datetime
from typing import Optional, Union

_EPS = 1e-8


def _parse_deadline(deadline, reference_dt: datetime) -> Optional[float]:
    """Return hours remaining until deadline, or None."""
    if deadline is None:
        return None
    if isinstance(deadline, str):
        deadline = datetime.fromisoformat(deadline)
    if isinstance(deadline, datetime):
        delta = (deadline - reference_dt).total_seconds() / 3600.0
        return delta
    if isinstance(deadline, (int, float)):
        return float(deadline)  # assume already in hours
    return None


class UrgencyLayer:
    """Produces 8D vector encoding deadline urgency at multiple scales."""

    dim = 8

    def encode(self, dt: datetime, context: Optional[dict] = None) -> np.ndarray:
        context = context or {}

        # Collect all deadlines
        deadlines_raw = context.get("deadlines", [])
        single = context.get("deadline")
        if single is not None and not deadlines_raw:
            deadlines_raw = [single]

        hours_remaining = []
        for d in deadlines_raw:
            h = _parse_deadline(d, dt)
            if h is not None:
                hours_remaining.append(h)

        if not hours_remaining:
            return np.zeros(self.dim, dtype=np.float32)

        # Primary deadline (most urgent)
        closest = min(hours_remaining, key=lambda h: abs(h))
        h = closest

        # Hyperbolic urgency at three discount rates
        # Slow (k=0.05): long-range awareness, deadline days away still registers
        slow_urgency = 1.0 / (1.0 + 0.05 * max(h, 0))
        # Medium (k=0.5): standard urgency curve
        med_urgency = 1.0 / (1.0 + 0.5 * max(h, 0))
        # Fast (k=5.0): last-minute panic, spikes within the final hour
        fast_urgency = 1.0 / (1.0 + 5.0 * max(h, 0))

        # Log time remaining (signed, captures both before and after deadline)
        log_remaining = np.sign(h) * np.log1p(abs(h)) / 10.0

        # Overdue features
        is_overdue = 1.0 / (1.0 + np.exp(h * 2.0))  # smooth sigmoid, ~1 when overdue
        overdue_severity = np.log1p(max(-h, 0)) / 5.0  # how far past deadline

        # Multi-deadline features
        n_deadlines = min(len(hours_remaining), 10) / 10.0
        # Aggregate urgency: max across all deadlines (medium rate)
        max_urgency = max(1.0 / (1.0 + 0.5 * max(hr, 0)) for hr in hours_remaining)

        features = np.array([
            slow_urgency,
            med_urgency,
            fast_urgency,
            log_remaining,
            is_overdue,
            overdue_severity,
            n_deadlines,
            max_urgency,
        ], dtype=np.float32)

        return features
