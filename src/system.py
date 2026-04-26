from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from src.recommender import load_songs
from src.rag.song_docs import build_song_doc
from src.rag.vector_store import InMemoryVectorStore
from src.rag.rag_recommender import RagSearchEngine
from src.agent.playlist_agent import PlaylistBuilderAgent


@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore:
    songs = load_songs("data/songs.csv")
    docs = [build_song_doc(s) for s in songs]
    store = InMemoryVectorStore()
    store.index(docs)
    return store


@lru_cache(maxsize=1)
def get_search_engine() -> RagSearchEngine:
    return RagSearchEngine(get_vector_store())


@lru_cache(maxsize=1)
def get_playlist_agent() -> PlaylistBuilderAgent:
    return PlaylistBuilderAgent(get_search_engine())


def load_user_profile_stub() -> Dict[str, object]:
    return {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
    }
