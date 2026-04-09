"""Temporal surprise encoding.

Neuroscience finding (Sohn et al., 2021): frontal cortex encodes
stimuli as deviations from temporal expectations. An event arriving
2 hours early or 3 days late carries information.

This layer encodes how surprising the current moment is relative
to temporal expectations -- either explicitly provided or estimated
from event history patterns.
"""

import numpy as np
from datetime import datetime
from typing import Optional

_EPS = 1e-8


class PredictionErrorLayer:
    """Produces 16D vector encoding temporal prediction error / surprise."""

    dim = 16

    def encode(self, dt: datetime, context: Optional[dict] = None) -> np.ndarray:
        context = context or {}
        features = np.zeros(self.dim, dtype=np.float32)

        t_expected = context.get("t_expected")
        event_history = context.get("event_history", [])

        # --- Explicit prediction error (8D) [indices 0-7] ---
        if t_expected is not None:
            if isinstance(t_expected, str):
                t_expected = datetime.fromisoformat(t_expected)
            if isinstance(t_expected, datetime):
                error_seconds = (dt - t_expected).total_seconds()
                error_hours = error_seconds / 3600.0

                # Multi-scale error features
                features[0] = np.tanh(error_seconds / 60.0)     # minute-scale
                features[1] = np.tanh(error_hours)               # hour-scale
                features[2] = np.tanh(error_hours / 24.0)        # day-scale
                features[3] = np.tanh(error_hours / 168.0)       # week-scale

                # Error magnitude (always positive, log-compressed)
                features[4] = float(np.log1p(abs(error_hours))) / 5.0

                # Error direction: smooth encoding
                features[5] = float(np.tanh(error_hours / 2.0))  # -1=early, +1=late

                # Surprise score: sigmoid of magnitude (caps at ~1)
                features[6] = float(2.0 / (1.0 + np.exp(-abs(error_hours))) - 1.0)

                # Binary: was this expected at all?
                features[7] = 1.0  # explicit expectation exists

        # --- History-derived prediction error (8D) [indices 8-15] ---
        if len(event_history) >= 3:
            # Parse event times
            times = []
            for event in event_history:
                if isinstance(event, datetime):
                    times.append(event.timestamp())
                elif isinstance(event, str):
                    try:
                        times.append(datetime.fromisoformat(event).timestamp())
                    except ValueError:
                        continue
                elif isinstance(event, (tuple, list)) and len(event) >= 1:
                    entry = event[0]
                    if isinstance(entry, datetime):
                        times.append(entry.timestamp())
                    elif isinstance(entry, str):
                        try:
                            times.append(datetime.fromisoformat(entry).timestamp())
                        except ValueError:
                            continue
                elif isinstance(event, dict) and "time" in event:
                    entry = event["time"]
                    if isinstance(entry, datetime):
                        times.append(entry.timestamp())
                    elif isinstance(entry, str):
                        try:
                            times.append(datetime.fromisoformat(entry).timestamp())
                        except ValueError:
                            continue

            if len(times) >= 3:
                times = sorted(times)
                intervals = np.diff(times)  # in seconds

                if len(intervals) >= 2:
                    mean_interval = float(np.mean(intervals))
                    std_interval = float(np.std(intervals))

                    # Expected next event time based on mean interval
                    last_event = times[-1]
                    expected_next = last_event + mean_interval
                    current_ts = dt.timestamp()
                    deviation = (current_ts - expected_next)  # seconds

                    # Normalized deviation
                    features[8] = np.tanh(deviation / (mean_interval + _EPS))

                    # Deviation in units of standard deviations
                    if std_interval > _EPS:
                        z_score = deviation / std_interval
                        features[9] = np.tanh(z_score / 3.0)  # normalized z-score
                    else:
                        features[9] = np.tanh(deviation / (mean_interval + _EPS))

                    # Regularity: coefficient of variation (low = regular, high = irregular)
                    cv = std_interval / (mean_interval + _EPS)
                    features[10] = float(1.0 / (1.0 + cv))  # 1 = perfectly regular, 0 = chaotic

                    # Burstiness (Goh-Barabási parameter)
                    features[11] = float((std_interval - mean_interval) / (std_interval + mean_interval + _EPS))

                    # Memory coefficient (correlation of consecutive intervals)
                    if len(intervals) >= 4:
                        i1 = intervals[:-1]
                        i2 = intervals[1:]
                        m1, m2 = np.mean(i1), np.mean(i2)
                        s1, s2 = np.std(i1), np.std(i2)
                        if s1 > _EPS and s2 > _EPS:
                            features[12] = float(np.mean((i1 - m1) * (i2 - m2)) / (s1 * s2))

                    # Trend: are events accelerating or decelerating?
                    if len(intervals) >= 4:
                        half = len(intervals) // 2
                        recent = float(np.mean(intervals[:half]))
                        older = float(np.mean(intervals[half:]))
                        features[13] = np.tanh((recent - older) / (older + _EPS))

                    # Period detection: dominant frequency via autocorrelation
                    if len(intervals) >= 6:
                        centered = intervals - np.mean(intervals)
                        autocorr = np.correlate(centered, centered, mode='full')
                        autocorr = autocorr[len(autocorr) // 2:]
                        if autocorr[0] > _EPS:
                            autocorr = autocorr / autocorr[0]
                            # Find first peak after lag 0
                            for lag in range(1, len(autocorr) - 1):
                                if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                                    dominant_period = lag * mean_interval
                                    # How well does current time align with this period?
                                    time_since_last = current_ts - times[-1]
                                    phase = (time_since_last % dominant_period) / dominant_period
                                    features[14] = float(np.cos(2.0 * np.pi * phase))  # phase alignment
                                    features[15] = float(autocorr[lag])  # strength of periodicity
                                    break

        return features
