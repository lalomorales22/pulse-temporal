"""Weber's Law temporal compression.

The felt difference between 1 min and 2 min is huge.
The felt difference between 101 min and 102 min is nothing.
Log transform captures this perceptual compression.
"""

import numpy as np
from datetime import datetime

# Reference epoch for normalization (2020-01-01)
_REFERENCE_EPOCH = datetime(2020, 1, 1).timestamp()
_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_WEEK = 604800.0
_SECONDS_PER_YEAR = 31557600.0  # 365.25 days


class LogTimeLayer:
    """Produces 8D vector encoding logarithmic time compression."""

    dim = 8

    def encode(self, dt: datetime) -> np.ndarray:
        ts = dt.timestamp()
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        day_of_week = dt.weekday()
        seconds_since_week_start = day_of_week * _SECONDS_PER_DAY + seconds_since_midnight
        day_of_year = dt.timetuple().tm_yday

        features = np.array([
            # Log-compressed absolute time (relative to reference epoch)
            np.log1p(max(ts - _REFERENCE_EPOCH, 0)) / 25.0,
            # Log-compressed time of day (seconds since midnight)
            np.log1p(seconds_since_midnight) / np.log1p(_SECONDS_PER_DAY),
            # Log-compressed time in week
            np.log1p(seconds_since_week_start) / np.log1p(_SECONDS_PER_WEEK),
            # Fractional hour (0-1, linear -- fine grain within the day)
            (dt.hour + dt.minute / 60.0) / 24.0,
            # Fractional week (0-1)
            (day_of_week + seconds_since_midnight / _SECONDS_PER_DAY) / 7.0,
            # Fractional year (0-1)
            (day_of_year - 1 + seconds_since_midnight / _SECONDS_PER_DAY) / 365.25,
            # Log-compressed day of year (captures Weber's law for seasonal position)
            np.log1p(day_of_year) / np.log1p(366),
            # Minute-level granularity (captures sub-hour patterns)
            dt.minute / 60.0,
        ], dtype=np.float32)

        return features
