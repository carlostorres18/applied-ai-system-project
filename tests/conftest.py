"""
Shared pytest fixtures.

Patches src.rag.vector_store.embed_texts with the hash-based fallback for
every test so no real OpenAI API calls are made during the test suite.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.rag.embeddings import EmbeddingResult, _hash_embed


def _hash_embed_texts(texts):
    text_list = list(texts)
    if not text_list:
        return EmbeddingResult(vectors=np.zeros((0, 512), dtype=np.float32))
    return EmbeddingResult(
        vectors=np.stack([_hash_embed(t) for t in text_list], axis=0)
    )


@pytest.fixture(autouse=True)
def no_openai_calls():
    with patch("src.rag.vector_store.embed_texts", side_effect=_hash_embed_texts):
        yield
