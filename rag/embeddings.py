"""Deterministic dense embeddings via the hashing trick.

Each token is hashed with MD5 (stable across processes, unlike Python's
salted ``hash()``) into one of ``dim`` buckets with a pseudo-random sign,
producing a signed count vector that is L2-normalized. Similar texts share
token buckets, so cosine similarity over these vectors behaves like a
(compact, lossy) bag-of-words embedding — enough to demonstrate dense
retrieval with zero external dependencies or API keys.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alphanumeric tokenization used across the platform."""
    return _TOKEN.findall(text.lower())


class HashEmbedding:
    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, text):
        """Return an L2-normalized dense vector (list of floats)."""
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


def cosine(a, b):
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
