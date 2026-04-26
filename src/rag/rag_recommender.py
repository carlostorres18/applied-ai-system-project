from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.rag.song_docs import SongDoc
from src.rag.vector_store import InMemoryVectorStore, SearchHit


@dataclass(frozen=True)
class RagResult:
    song: Dict[str, Any]
    retrieval_score: float
    evidence: Dict[str, Any]
    explanation: str


class RagSearchEngine:
    def __init__(self, store: InMemoryVectorStore):
        self.store = store

    def search(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RagResult]:
        expanded_query = _expand_query(query, user_profile)
        hits = self.store.search(expanded_query, k=max(k * 3, 10), filters=filters)
        reranked = _rerank(hits, user_profile)

        results: List[RagResult] = []
        for hit in reranked[:k]:
            evidence = _build_evidence(hit.doc)
            explanation = _explain(query=query, user_profile=user_profile, doc=hit.doc, evidence=evidence)
            results.append(
                RagResult(
                    song=dict(hit.doc.metadata),
                    retrieval_score=hit.score,
                    evidence=evidence,
                    explanation=explanation,
                )
            )

        return results


def _expand_query(query: str, user_profile: Optional[Dict[str, Any]]) -> str:
    if not user_profile:
        return query

    extras: List[str] = []
    genre = user_profile.get("favorite_genre") or user_profile.get("genre")
    mood = user_profile.get("favorite_mood") or user_profile.get("mood")

    if genre:
        extras.append(f"preferred genre: {genre}")
    if mood:
        extras.append(f"preferred mood: {mood}")

    if not extras:
        return query

    return query + "\n" + "\n".join(extras)


def _rerank(hits: Sequence[SearchHit], user_profile: Optional[Dict[str, Any]]) -> List[SearchHit]:
    if not user_profile:
        return list(hits)

    fav_genre = str(user_profile.get("favorite_genre") or user_profile.get("genre") or "").lower()
    fav_mood = str(user_profile.get("favorite_mood") or user_profile.get("mood") or "").lower()
    target_energy = user_profile.get("target_energy")

    def boost(hit: SearchHit) -> float:
        score = hit.score
        genre = str(hit.doc.metadata.get("genre", "")).lower()
        mood = str(hit.doc.metadata.get("mood", "")).lower()
        energy = hit.doc.metadata.get("energy")

        if fav_genre and genre == fav_genre:
            score += 0.15
        if fav_mood and mood == fav_mood:
            score += 0.10
        if target_energy is not None and energy is not None:
            delta = abs(float(energy) - float(target_energy))
            score += max(0.0, 0.10 - (delta * 0.10))

        return score

    return sorted(hits, key=boost, reverse=True)


def _build_evidence(doc: SongDoc) -> Dict[str, Any]:
    keys = [
        "title",
        "artist",
        "genre",
        "mood",
        "energy",
        "tempo_bpm",
        "valence",
        "danceability",
        "acousticness",
    ]
    return {k: doc.metadata.get(k) for k in keys}


def _explain(
    query: str,
    user_profile: Optional[Dict[str, Any]],
    doc: SongDoc,
    evidence: Dict[str, Any],
) -> str:
    bits: List[str] = []
    bits.append(f"Matched your search: '{query}'")

    if user_profile:
        genre = user_profile.get("favorite_genre") or user_profile.get("genre")
        mood = user_profile.get("favorite_mood") or user_profile.get("mood")
        if genre and str(evidence.get("genre", "")).lower() == str(genre).lower():
            bits.append(f"genre aligns with your taste ({evidence.get('genre')})")
        if mood and str(evidence.get("mood", "")).lower() == str(mood).lower():
            bits.append(f"mood aligns with your taste ({evidence.get('mood')})")

    bits.append(f"energy={evidence.get('energy')}, tempo_bpm={evidence.get('tempo_bpm')}")
    return "; ".join(bits)
