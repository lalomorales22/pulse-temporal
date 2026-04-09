"""Biological clock encoding.

The brain's suprachiasmatic nucleus drives ~24h circadian rhythms.
The Basic Rest-Activity Cycle adds ~90min ultradian oscillations.
Cognitive performance peaks roughly 10am-12pm and 4pm-6pm,
with a post-lunch dip around 1pm-3pm and lowest point at 3am-5am.

This layer encodes circadian phase and estimated cognitive state.
"""

import numpy as np
from datetime import datetime

_TWO_PI = 2.0 * np.pi

# Circadian cognitive performance curve (approximate, based on research)
# Maps hour-of-day to relative cognitive capacity [0, 1]
# Based on aggregated circadian performance data
_COGNITIVE_CURVE = np.array([
    # 0am  1am   2am   3am   4am   5am   6am   7am   8am   9am  10am  11am
    0.20, 0.15, 0.12, 0.10, 0.12, 0.18, 0.30, 0.50, 0.70, 0.85, 0.95, 0.92,
    # 12pm  1pm  2pm   3pm   4pm   5pm   6pm   7pm   8pm   9pm  10pm  11pm
    0.85, 0.72, 0.68, 0.75, 0.88, 0.90, 0.82, 0.70, 0.55, 0.40, 0.30, 0.25,
], dtype=np.float32)

# Energy curve (slightly different from cognitive -- physical energy)
_ENERGY_CURVE = np.array([
    0.15, 0.10, 0.08, 0.07, 0.10, 0.20, 0.40, 0.60, 0.75, 0.85, 0.90, 0.88,
    0.82, 0.70, 0.65, 0.72, 0.85, 0.88, 0.80, 0.65, 0.50, 0.35, 0.25, 0.18,
], dtype=np.float32)


def _interpolate_curve(curve: np.ndarray, hour_float: float) -> float:
    """Smooth interpolation on a 24-point curve."""
    h0 = int(hour_float) % 24
    h1 = (h0 + 1) % 24
    frac = hour_float - int(hour_float)
    return float(curve[h0] * (1.0 - frac) + curve[h1] * frac)


class CircadianLayer:
    """Produces 8D vector encoding circadian and ultradian phase."""

    dim = 8

    def encode(self, dt: datetime) -> np.ndarray:
        hour_float = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        minutes_of_day = dt.hour * 60 + dt.minute + dt.second / 60.0

        features = np.array([
            # Primary circadian cycle (24h) -- sin/cos
            np.sin(_TWO_PI * hour_float / 24.0),
            np.cos(_TWO_PI * hour_float / 24.0),
            # First harmonic (12h) -- captures AM/PM asymmetry
            np.sin(_TWO_PI * hour_float / 12.0),
            np.cos(_TWO_PI * hour_float / 12.0),
            # Ultradian cycle (~90 min BRAC)
            np.sin(_TWO_PI * minutes_of_day / 90.0),
            np.cos(_TWO_PI * minutes_of_day / 90.0),
            # Estimated cognitive performance (interpolated from research curve)
            _interpolate_curve(_COGNITIVE_CURVE, hour_float),
            # Estimated energy level
            _interpolate_curve(_ENERGY_CURVE, hour_float),
        ], dtype=np.float32)

        return features
