from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.system import get_playlist_agent, get_search_engine


app = FastAPI(title="Music Recommender (RAG + Agent)")


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    user_profile: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    song: Dict[str, Any]
    retrieval_score: float
    evidence: Dict[str, Any]
    explanation: str


class PlaylistRequest(BaseModel):
    goal: str
    minutes: int = 45
    user_profile: Optional[Dict[str, Any]] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PlaylistItem(BaseModel):
    song: Dict[str, Any]
    explanation: str


class PlaylistResponse(BaseModel):
    title: str
    goal: str
    minutes: int
    items: List[PlaylistItem]
    trace: List[str]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    engine = get_search_engine()
    results = engine.search(
        req.query,
        user_profile=req.user_profile,
        k=req.k,
        filters=req.filters,
    )
    return [SearchResult(**r.__dict__) for r in results]


@app.post("/playlist", response_model=PlaylistResponse)
def playlist(req: PlaylistRequest) -> PlaylistResponse:
    agent = get_playlist_agent()
    plan = agent.build_playlist(
        goal=req.goal,
        user_profile=req.user_profile,
        minutes=req.minutes,
        constraints=req.constraints,
    )
    return PlaylistResponse(
        title=plan.title,
        goal=plan.goal,
        minutes=plan.minutes,
        items=[PlaylistItem(song=i.song, explanation=i.explanation) for i in plan.items],
        trace=plan.trace,
    )
