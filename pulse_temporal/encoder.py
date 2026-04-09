"""PulseEncoder: the main embedding model.

Combines all seven temporal signal layers into a single fixed-dimensional
embedding vector where vector distance approximates felt temporal distance.

Usage:
    from pulse_temporal import PulseEncoder
    pulse = PulseEncoder()
    emb = pulse.encode("2026-04-09T14:30:00", context={"deadline": "2026-04-09T17:00:00"})
"""

import numpy as np
from datetime import datetime
from typing import Optional, Union, List, Tuple

from .layers import (
    LogTimeLayer,
    OscillatorLayer,
    CircadianLayer,
    CalendarLayer,
    UrgencyLayer,
    TemporalStateLayer,
    PredictionErrorLayer,
)


class PulseEncoder:
    """Encodes moments as rich temporal embedding vectors.

    Each moment is encoded by fusing seven signal layers:
        1. log_time       (8D)  - Weber's Law logarithmic compression
        2. oscillators    (32D) - Multi-frequency sinusoids (Time2Vec base)
        3. circadian       (8D) - 24h + 90min biological clock phase
        4. calendar       (24D) - Structural calendar features
        5. urgency         (8D) - Hyperbolic deadline proximity
        6. temporal_state (32D) - Continuous-time event history state
        7. prediction_error(16D) - Temporal surprise encoding

    Total raw dimension: 128D (default output dimension).
    """

    # Per-layer importance weights. Context-dependent layers (urgency,
    # temporal_state, prediction_error) get higher weights so they
    # meaningfully shift the embedding when context is provided.
    # In v0.3+ these become learned parameters.
    _DEFAULT_LAYER_WEIGHTS = {
        "log_time": 1.0,
        "oscillators": 0.5,       # dense 32D, scale down to prevent dominance
        "circadian": 1.5,         # biologically important signal
        "calendar": 0.6,          # structural but less experiential
        "urgency": 4.0,           # critical experiential signal when present
        "temporal_state": 2.0,    # rich context when event history exists
        "prediction_error": 3.0,  # surprise is a strong experiential signal
    }

    def __init__(self, model_name: Optional[str] = None, dim: int = 128,
                 layer_weights: Optional[dict] = None):
        self.dim = dim
        self.model_name = model_name or "pulse-base-v1"

        # Initialize layers
        self._log_time = LogTimeLayer()
        self._oscillators = OscillatorLayer()
        self._circadian = CircadianLayer()
        self._calendar = CalendarLayer()
        self._urgency = UrgencyLayer()
        self._temporal_state = TemporalStateLayer()
        self._prediction_error = PredictionErrorLayer()

        self._layer_weights = layer_weights or self._DEFAULT_LAYER_WEIGHTS

        self._raw_dim = (
            self._log_time.dim
            + self._oscillators.dim
            + self._circadian.dim
            + self._calendar.dim
            + self._urgency.dim
            + self._temporal_state.dim
            + self._prediction_error.dim
        )  # 128

        # Projection matrix for dimension adjustment (if needed)
        if self._raw_dim != dim:
            rng = np.random.RandomState(seed=2026)
            raw = rng.randn(self._raw_dim, dim).astype(np.float32)
            self._projection, _ = np.linalg.qr(raw)
            if self._projection.shape[1] < dim:
                pad = np.zeros((self._raw_dim, dim - self._projection.shape[1]), dtype=np.float32)
                self._projection = np.hstack([self._projection, pad])
        else:
            self._projection = None

    def _parse_time(self, t: Union[str, datetime, float, int]) -> datetime:
        if isinstance(t, datetime):
            return t
        if isinstance(t, str):
            return datetime.fromisoformat(t)
        if isinstance(t, (int, float)):
            return datetime.fromtimestamp(t)
        raise TypeError(f"Cannot parse time from {type(t)}: {t}")

    def encode(
        self,
        t: Union[str, datetime, float, int],
        context: Optional[dict] = None,
    ) -> np.ndarray:
        """Encode a single moment as a temporal embedding vector.

        Args:
            t: Timestamp as ISO string, datetime, or unix timestamp.
            context: Optional dict with keys like:
                - deadline: str/datetime for primary deadline
                - deadlines: list of deadlines
                - events_today: int, number of events/meetings today
                - sleep_hours: float, hours of sleep last night
                - hours_active: float, hours since waking
                - event_history: list of past event timestamps
                - t_expected: str/datetime of when this event was expected
                - mood: str (not used in v0.1, reserved for arousal modulation)

        Returns:
            numpy array of shape (dim,) -- the temporal embedding.
        """
        dt = self._parse_time(t)
        context = context or {}

        # Compute each layer with importance weighting
        w = self._layer_weights
        parts = [
            self._log_time.encode(dt) * w["log_time"],
            self._oscillators.encode(dt) * w["oscillators"],
            self._circadian.encode(dt) * w["circadian"],
            self._calendar.encode(dt) * w["calendar"],
            self._urgency.encode(dt, context) * w["urgency"],
            self._temporal_state.encode(dt, context) * w["temporal_state"],
            self._prediction_error.encode(dt, context) * w["prediction_error"],
        ]

        raw = np.concatenate(parts)

        # Apply projection if dimensions don't match
        if self._projection is not None:
            raw = raw @ self._projection

        # L2 normalize for cosine similarity compatibility
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm

        return raw.astype(np.float32)

    def encode_batch(
        self,
        timestamps: List[Union[str, datetime, float, int]],
        contexts: Optional[List[Optional[dict]]] = None,
    ) -> np.ndarray:
        """Encode multiple moments. Returns array of shape (n, dim)."""
        if contexts is None:
            contexts = [None] * len(timestamps)
        embeddings = [self.encode(t, c) for t, c in zip(timestamps, contexts)]
        return np.stack(embeddings)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings. Range: [-1, 1]."""
        return float(np.dot(emb1, emb2))

    def similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Pairwise cosine similarity matrix."""
        mat = np.stack(embeddings)
        return mat @ mat.T

    def distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Euclidean distance between two embeddings."""
        return float(np.linalg.norm(emb1 - emb2))

    def temporal_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Felt temporal distance: 1 - cosine_similarity, range [0, 2]."""
        return 1.0 - self.similarity(emb1, emb2)

    def decompose(
        self,
        t: Union[str, datetime, float, int],
        context: Optional[dict] = None,
    ) -> dict:
        """Return the raw (un-normalized) output of each layer for inspection."""
        dt = self._parse_time(t)
        context = context or {}
        return {
            "log_time": self._log_time.encode(dt),
            "oscillators": self._oscillators.encode(dt),
            "circadian": self._circadian.encode(dt),
            "calendar": self._calendar.encode(dt),
            "urgency": self._urgency.encode(dt, context),
            "temporal_state": self._temporal_state.encode(dt, context),
            "prediction_error": self._prediction_error.encode(dt, context),
        }

    def get_temporal_context(
        self,
        t: Optional[Union[str, datetime]] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Return a rich temporal context package for LLM injection.

        This is the primary interface for the PULSE/MIND architecture:
        the MIND queries this before every response.
        """
        if t is None:
            t = datetime.now()
        dt = self._parse_time(t)
        context = context or {}

        embedding = self.encode(dt, context)
        decomposed = self.decompose(dt, context)

        # Circadian phase name
        hour = dt.hour
        if 6 <= hour < 10:
            phase = "morning_ramp"
        elif 10 <= hour < 12:
            phase = "morning_peak"
        elif 12 <= hour < 14:
            phase = "post_lunch_dip"
        elif 14 <= hour < 17:
            phase = "afternoon_peak"
        elif 17 <= hour < 20:
            phase = "evening_wind_down"
        elif 20 <= hour < 23:
            phase = "night_transition"
        else:
            phase = "deep_night"

        # Urgency summary
        urgency_vec = decomposed["urgency"]
        max_urgency = float(urgency_vec[1])  # medium-rate urgency
        if max_urgency > 0.8:
            urgency_level = "critical"
        elif max_urgency > 0.5:
            urgency_level = "high"
        elif max_urgency > 0.2:
            urgency_level = "moderate"
        elif max_urgency > 0.01:
            urgency_level = "low"
        else:
            urgency_level = "none"

        # Cognitive estimate from circadian layer
        cognitive = float(decomposed["circadian"][6])
        energy = float(decomposed["circadian"][7])

        return {
            "embedding": embedding,
            "timestamp": dt.isoformat(),
            "circadian_phase": phase,
            "cognitive_capacity": round(cognitive, 2),
            "energy_level": round(energy, 2),
            "urgency_level": urgency_level,
            "urgency_score": round(max_urgency, 3),
        }

    def __repr__(self) -> str:
        return f"PulseEncoder(model='{self.model_name}', dim={self.dim})"
