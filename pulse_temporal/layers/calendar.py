"""Structural time features.

Tuesday has a different vibe than Friday. December feels different from July.
This layer captures the structural/cultural properties of calendar time
using cyclical encodings and categorical features.
"""

import numpy as np
from datetime import datetime, date

_TWO_PI = 2.0 * np.pi

# Major US holidays (month, day) -- expandable
_US_HOLIDAYS = [
    (1, 1),    # New Year's Day
    (1, 15),   # MLK Day (approx)
    (2, 14),   # Valentine's Day
    (2, 19),   # Presidents' Day (approx)
    (3, 17),   # St. Patrick's Day
    (5, 27),   # Memorial Day (approx)
    (6, 19),   # Juneteenth
    (7, 4),    # Independence Day
    (9, 2),    # Labor Day (approx)
    (10, 31),  # Halloween
    (11, 11),  # Veterans Day
    (11, 28),  # Thanksgiving (approx)
    (12, 24),  # Christmas Eve
    (12, 25),  # Christmas Day
    (12, 31),  # New Year's Eve
]


def _days_to_nearest_holiday(dt: datetime) -> float:
    """Minimum days to any holiday in current year (wrapping around year boundary)."""
    current_doy = dt.timetuple().tm_yday
    year = dt.year
    min_dist = 366.0

    for month, day in _US_HOLIDAYS:
        try:
            hol = date(year, month, day)
        except ValueError:
            continue
        hol_doy = hol.timetuple().tm_yday
        dist = abs(current_doy - hol_doy)
        dist = min(dist, 365 - dist)  # wrap around year
        min_dist = min(min_dist, dist)

    return min_dist


def _season_encoding(day_of_year: int) -> np.ndarray:
    """Smooth 4D season encoding using overlapping Gaussian-like activations."""
    # Season peaks (day of year): spring=80, summer=172, fall=266, winter=355
    peaks = np.array([80, 172, 266, 355], dtype=np.float64)
    sigma = 45.0  # width of each season's activation
    dists = np.minimum(
        np.abs(day_of_year - peaks),
        365.0 - np.abs(day_of_year - peaks)
    )
    activations = np.exp(-0.5 * (dists / sigma) ** 2)
    # Normalize to sum to 1
    return (activations / activations.sum()).astype(np.float32)


def _time_period_encoding(hour: float) -> np.ndarray:
    """Soft 4D encoding: morning (6-12), afternoon (12-17), evening (17-22), night (22-6)."""
    # Use overlapping Gaussian activations centered at each period's midpoint
    centers = np.array([9.0, 14.5, 19.5, 3.0])  # midpoints
    sigma = 3.0
    dists = np.minimum(
        np.abs(hour - centers),
        24.0 - np.abs(hour - centers)
    )
    activations = np.exp(-0.5 * (dists / sigma) ** 2)
    return (activations / activations.sum()).astype(np.float32)


class CalendarLayer:
    """Produces 24D vector of structural calendar features."""

    dim = 24

    def encode(self, dt: datetime) -> np.ndarray:
        dow = dt.weekday()  # 0=Monday
        month = dt.month    # 1-12
        dom = dt.day        # 1-31
        doy = dt.timetuple().tm_yday
        woy = dt.isocalendar()[1]
        hour_float = dt.hour + dt.minute / 60.0
        is_weekend = float(dow >= 5)

        holiday_dist = _days_to_nearest_holiday(dt)
        is_holiday = float(holiday_dist < 1.0)
        holiday_proximity = float(np.exp(-holiday_dist / 3.0))  # decay with ~3 day half-life

        is_business = float(not is_weekend and 9 <= dt.hour < 17)
        year_progress = (doy - 1) / 365.25

        # Month boundary proximity (distance to 1st or last day)
        days_from_start = dom - 1
        # Approximate days from month end
        if month in (1, 3, 5, 7, 8, 10, 12):
            days_in_month = 31
        elif month in (4, 6, 9, 11):
            days_in_month = 30
        else:
            days_in_month = 29 if (dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0)) else 28
        days_from_end = days_in_month - dom
        month_boundary = float(np.exp(-min(days_from_start, days_from_end) / 2.0))

        features = np.concatenate([
            # Cyclical encodings (2D each)
            np.array([np.sin(_TWO_PI * dow / 7.0), np.cos(_TWO_PI * dow / 7.0)]),
            np.array([np.sin(_TWO_PI * (month - 1) / 12.0), np.cos(_TWO_PI * (month - 1) / 12.0)]),
            np.array([np.sin(_TWO_PI * (dom - 1) / 31.0), np.cos(_TWO_PI * (dom - 1) / 31.0)]),
            np.array([np.sin(_TWO_PI * woy / 52.0), np.cos(_TWO_PI * woy / 52.0)]),
            np.array([np.sin(_TWO_PI * doy / 366.0), np.cos(_TWO_PI * doy / 366.0)]),
            # Binary/scalar features
            np.array([is_weekend, is_holiday, holiday_proximity]),
            # Season (4D soft encoding)
            _season_encoding(doy),
            # Time of day period (4D soft encoding)
            _time_period_encoding(hour_float),
            # Business/structural
            np.array([is_business, year_progress, month_boundary]),
        ]).astype(np.float32)

        return features
