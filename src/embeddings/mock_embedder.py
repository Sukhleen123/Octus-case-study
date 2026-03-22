"""
MockEmbedder — deterministic random vectors for development / testing.

Vectors are seeded from the text hash, so the same text always produces
the same vector (stable across runs). This enables FAISS indexing without
any API calls.
"""

from __future__ import annotations

import hashlib
import math
import struct

from src.embeddings.embedder import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """
    Returns deterministic pseudo-random unit vectors.

    Not semantically meaningful — use only for smoke tests and
    offline development where embedding quality doesn't matter.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Deterministic vector from SHA-256 of text."""
        digest = hashlib.sha256(text.encode()).digest()
        # Expand digest into `dim` floats via repeated hashing
        floats: list[float] = []
        block = digest
        while len(floats) < self.dim:
            # Each 4 bytes -> 1 float in [-1, 1]
            for i in range(0, len(block) - 3, 4):
                val = struct.unpack_from(">i", block, i)[0]
                floats.append(val / 2**31)
            block = hashlib.sha256(block).digest()

        raw = floats[: self.dim]
        # L2-normalize so cosine similarity works correctly
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
