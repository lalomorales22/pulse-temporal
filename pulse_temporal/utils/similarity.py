"""Similarity and distance functions for temporal embeddings."""

import numpy as np
from typing import List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Assumes L2-normalized inputs."""
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def temporal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Felt temporal distance: 1 - cosine_similarity. Range [0, 2]."""
    return 1.0 - cosine_similarity(a, b)


def similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """Pairwise cosine similarity matrix for a list of embeddings."""
    mat = np.stack(embeddings)
    return mat @ mat.T
