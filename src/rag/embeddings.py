from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from openai import AuthenticationError, BadRequestError

from src.rag.openai_client import get_openai_client


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray


def embed_texts(texts: Iterable[str], model: str = "text-embedding-3-small") -> EmbeddingResult:
    text_list = list(texts)
    if not text_list:
        return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32))

    try:
        client = get_openai_client()
        response = client.embeddings.create(model=model, input=text_list)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return EmbeddingResult(vectors=vectors)
    except (AuthenticationError, BadRequestError):
        vectors = np.stack([_hash_embed(t) for t in text_list], axis=0)
        return EmbeddingResult(vectors=vectors)


def _hash_embed(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for token in text.lower().split():
        h = hash(token) % dim
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query = query_vec.astype(np.float32)
    docs = doc_vecs.astype(np.float32)

    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)

    denom = (doc_norms * query_norm) + 1e-12
    sims = (docs @ query) / denom
    return sims
