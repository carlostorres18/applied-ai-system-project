from __future__ import annotations

import json
from typing import Any, Dict, Optional

from omniagents import function_tool

from src.system import get_playlist_agent, get_search_engine, load_user_profile_stub


@function_tool
def search_songs(query: str, k: int = 5, user_profile_json: Optional[str] = None) -> str:
    """Search the song catalog using natural language.

    Args:
        query: Natural-language description of desired songs.
        k: Number of results to return.
        user_profile_json: Optional JSON string with user taste preferences.

    Returns:
        JSON string with ranked results including evidence and explanations.
    """

    user_profile: Dict[str, Any]
    if user_profile_json:
        user_profile = json.loads(user_profile_json)
    else:
        user_profile = load_user_profile_stub()

    engine = get_search_engine()
    results = engine.search(query, user_profile=user_profile, k=int(k))
    return json.dumps([r.__dict__ for r in results])


@function_tool
def build_playlist(
    goal: str,
    minutes: int = 45,
    constraints_json: Optional[str] = None,
    user_profile_json: Optional[str] = None,
) -> str:
    """Build a playlist to match a user's goal.

    Args:
        goal: Natural-language playlist goal (vibe, activity, etc.).
        minutes: Desired playlist duration in minutes.
        constraints_json: Optional JSON constraints (e.g., {"unique_artist": true}).
        user_profile_json: Optional JSON string with user taste preferences.

    Returns:
        JSON string with playlist plan, trace, and per-song explanations.
    """

    user_profile: Dict[str, Any]
    if user_profile_json:
        user_profile = json.loads(user_profile_json)
    else:
        user_profile = load_user_profile_stub()

    constraints: Dict[str, Any] = json.loads(constraints_json) if constraints_json else {}

    agent = get_playlist_agent()
    plan = agent.build_playlist(
        goal=goal,
        user_profile=user_profile,
        minutes=int(minutes),
        constraints=constraints,
    )

    payload = {
        "title": plan.title,
        "goal": plan.goal,
        "minutes": plan.minutes,
        "trace": plan.trace,
        "items": [{"song": i.song, "explanation": i.explanation} for i in plan.items],
    }
    return json.dumps(payload)
