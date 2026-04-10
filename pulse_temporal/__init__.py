"""pulse-temporal: Experiential time embeddings for AI.

Not timestamps -- time.

Usage:
    from pulse_temporal import PulseEncoder
    pulse = PulseEncoder()
    emb = pulse.encode("2026-04-09T14:30:00", context={"deadline": "2026-04-09T17:00:00"})

Middleware:
    from pulse_temporal import PulseMiddleware
    mw = PulseMiddleware()
    system_prompt = mw.get_temporal_system_prompt()
"""

from .encoder import PulseEncoder
from .utils.similarity import cosine_similarity, temporal_distance, similarity_matrix
from .middleware import PulseMiddleware

__version__ = "0.2.0"
__all__ = [
    "PulseEncoder",
    "PulseMiddleware",
    "cosine_similarity",
    "temporal_distance",
    "similarity_matrix",
]
