"""Multi-frequency learned sinusoids (Time2Vec base).

Inspired by the Striatal Beat Frequency model from neuroscience:
time is coded through coincidental activation of neurons oscillating
at different frequencies. Specific beat patterns = specific durations.

For v0.1, frequencies are fixed at meaningful time periods.
In v0.3+, they become learnable parameters trained end-to-end.
"""

import numpy as np
from datetime import datetime

# Periods in seconds covering human-relevant time scales
_PERIODS_SECONDS = np.array([
    3600,       # 1 hour
    7200,       # 2 hours
    14400,      # 4 hours
    21600,      # 6 hours
    28800,      # 8 hours (work day / sleep cycle)
    43200,      # 12 hours
    86400,      # 24 hours (daily)
    172800,     # 2 days
    604800,     # 1 week
    1209600,    # 2 weeks
    2592000,    # ~30 days (monthly)
    7776000,    # ~90 days (quarterly)
    15552000,   # ~180 days (half year)
    31557600,   # ~365.25 days (yearly)
    63115200,   # ~2 years
    126230400,  # ~4 years
], dtype=np.float64)

_ANGULAR_FREQUENCIES = 2.0 * np.pi / _PERIODS_SECONDS  # ω = 2π/T

# Fixed phase offsets (seeded for reproducibility)
_rng = np.random.RandomState(42)
_PHASE_OFFSETS = _rng.uniform(0, 2 * np.pi, size=len(_PERIODS_SECONDS))

# Reference epoch
_REFERENCE_EPOCH = datetime(2020, 1, 1).timestamp()


class OscillatorLayer:
    """Produces 32D vector of multi-frequency sin/cos oscillations."""

    dim = 32

    def encode(self, dt: datetime) -> np.ndarray:
        t = dt.timestamp() - _REFERENCE_EPOCH

        # sin(ωt + φ) and cos(ωt + φ) for each frequency
        angles = _ANGULAR_FREQUENCIES * t + _PHASE_OFFSETS
        sines = np.sin(angles)
        cosines = np.cos(angles)

        # Interleave: [sin_1, cos_1, sin_2, cos_2, ...]
        features = np.empty(32, dtype=np.float32)
        features[0::2] = sines
        features[1::2] = cosines

        return features
