from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.rag.rag_recommender import RagResult, RagSearchEngine


@dataclass(frozen=True)
class PlaylistItem:
    song: Dict[str, Any]
    explanation: str


@dataclass(frozen=True)
class PlaylistPlan:
    title: str
    goal: str
    minutes: int
    items: List[PlaylistItem]
    trace: List[str]


class PlaylistBuilderAgent:
    def __init__(self, search_engine: RagSearchEngine):
        self.search_engine = search_engine

    def build_playlist(
        self,
        goal: str,
        user_profile: Optional[Dict[str, Any]] = None,
        minutes: int = 45,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> PlaylistPlan:
        constraints = constraints or {}
        trace: List[str] = []

        subqueries = _make_subqueries(goal)
        trace.append(f"subqueries={subqueries}")

        candidates: List[RagResult] = []
        for q in subqueries:
            filters = _filters_from_constraints(constraints)
            hits = self.search_engine.search(q, user_profile=user_profile, k=10, filters=filters)
            candidates.extend(hits)
        trace.append(f"candidates={len(candidates)}")

        ranked = _rank_candidates(candidates, user_profile=user_profile, constraints=constraints)
        trace.append(f"ranked={len(ranked)}")

        target_count = max(5, int(round(minutes / 3.5)))
        trace.append(f"target_count={target_count}")

        picked = _assemble_playlist(ranked, target_count=target_count, constraints=constraints)
        trace.append(f"picked={len(picked)}")

        title = _make_title(goal)
        items = [PlaylistItem(song=p.song, explanation=p.explanation) for p in picked]

        return PlaylistPlan(title=title, goal=goal, minutes=minutes, items=items, trace=trace)


def _make_subqueries(goal: str) -> List[str]:
    g = goal.strip()
    if not g:
        return ["popular songs"]

    seeds = [g]
    lower = g.lower()

    if "study" in lower or "focus" in lower or "coding" in lower:
        seeds.append(g + ", chill lofi, instrumental")
        seeds.append(g + ", low energy, mellow")
    elif "workout" in lower or "gym" in lower or "run" in lower:
        seeds.append(g + ", high energy, intense")
        seeds.append(g + ", upbeat, danceable")
    elif "sleep" in lower:
        seeds.append(g + ", ambient, calm")
        seeds.append(g + ", low tempo")

    return list(dict.fromkeys(seeds))


def _filters_from_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if "genre" in constraints and constraints["genre"]:
        filters["genre"] = constraints["genre"]
    if "mood" in constraints and constraints["mood"]:
        filters["mood"] = constraints["mood"]
    return filters


def _rank_candidates(
    candidates: Sequence[RagResult],
    user_profile: Optional[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> List[RagResult]:
    fav_genre = str((user_profile or {}).get("favorite_genre") or (user_profile or {}).get("genre") or "").lower()
    fav_mood = str((user_profile or {}).get("favorite_mood") or (user_profile or {}).get("mood") or "").lower()

    def score(item: RagResult) -> float:
        s = item.retrieval_score
        genre = str(item.song.get("genre", "")).lower()
        mood = str(item.song.get("mood", "")).lower()
        if fav_genre and genre == fav_genre:
            s += 0.1
        if fav_mood and mood == fav_mood:
            s += 0.1
        return s

    return sorted(candidates, key=score, reverse=True)


def _assemble_playlist(
    ranked: Sequence[RagResult],
    target_count: int,
    constraints: Dict[str, Any],
) -> List[RagResult]:
    seen_song_ids: set[int] = set()
    seen_artists: set[str] = set()
    enforce_unique_artist = bool(constraints.get("unique_artist", True))

    picked: List[RagResult] = []
    for item in ranked:
        song_id = int(item.song.get("id"))
        artist = str(item.song.get("artist", ""))

        if song_id in seen_song_ids:
            continue
        if enforce_unique_artist and artist and artist in seen_artists:
            continue

        picked.append(item)
        seen_song_ids.add(song_id)
        if artist:
            seen_artists.add(artist)

        if len(picked) >= target_count:
            break

    return picked


def _make_title(goal: str) -> str:
    g = goal.strip()
    if not g:
        return "Your Playlist"
    return "Playlist: " + g[:48]
