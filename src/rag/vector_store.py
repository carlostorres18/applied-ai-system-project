from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.rag.embeddings import cosine_sim_matrix, embed_texts
from src.rag.song_docs import SongDoc


@dataclass
class SearchHit:
    doc: SongDoc
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._docs: List[SongDoc] = []
        self._vectors: Optional[np.ndarray] = None

    @property
    def docs(self) -> Sequence[SongDoc]:
        return self._docs

    def index(self, docs: Iterable[SongDoc]) -> None:
        doc_list = list(docs)
        vectors = embed_texts([d.text for d in doc_list]).vectors
        self._docs = doc_list
        self._vectors = vectors

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        if self._vectors is None or not self._docs:
            raise RuntimeError("Vector store is empty. Call index() first.")

        filters = filters or {}
        query_vec = embed_texts([query]).vectors[0]

        doc_indices = [i for i, d in enumerate(self._docs) if _passes_filters(d, filters)]
        if not doc_indices:
            return []

        doc_vecs = self._vectors[doc_indices]
        sims = cosine_sim_matrix(query_vec, doc_vecs)

        top_local = np.argsort(-sims)[:k]
        hits: List[SearchHit] = []
        for local_rank in top_local:
            global_idx = doc_indices[int(local_rank)]
            hits.append(SearchHit(doc=self._docs[global_idx], score=float(sims[int(local_rank)])))
        return hits


def _passes_filters(doc: SongDoc, filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        value = doc.metadata.get(key)
        if expected is None:
            continue
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        else:
            if value != expected:
                return False
    return True
