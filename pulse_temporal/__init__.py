"""pulse-temporal: Experiential time embeddings for AI.

Not timestamps -- time.

Usage:
    from pulse_temporal import PulseEncoder
    pulse = PulseEncoder()
    emb = pulse.encode("2026-04-09T14:30:00", context={"deadline": "2026-04-09T17:00:00"})
"""

from .encoder import PulseEncoder
from .utils.similarity import cosine_similarity, temporal_distance, similarity_matrix

__version__ = "0.1.0"
__all__ = [
    "PulseEncoder",
    "cosine_similarity",
    "temporal_distance",
    "similarity_matrix",
]
