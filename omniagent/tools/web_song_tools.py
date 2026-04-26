from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from omniagents import function_tool
from urllib.parse import urlparse, urlunparse

from src.rag.env import load_env


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _canonical_url(url: str) -> str:
    """Strip query params and fragment so the same track with different tracking params is caught."""
    try:
        p = urlparse(url)
        return urlunparse(p._replace(query="", fragment="")).rstrip("/")
    except Exception:
        return url


def _canonical_title(title: str) -> str:
    return title.lower().strip()


def _get_serpapi_key() -> str:
    load_env()
    return os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY") or ""


def web_search_songs_impl(query: str, k: int = 8, source: str = "spotify") -> str:
    api_key = _get_serpapi_key()
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY (or SERPAPI_KEY) in environment")

    source = (source or "general").strip().lower()

    site_filter = ""
    if source == "spotify":
        site_filter = "site:open.spotify.com track"
    elif source in {"apple", "apple_music"}:
        site_filter = "site:music.apple.com song"
    elif source == "genius":
        site_filter = "site:genius.com lyrics"
    elif source == "youtube":
        site_filter = "site:youtube.com"

    q = f"{site_filter} {query}".strip()

    params = {
        "engine": "google",
        "q": q,
        "api_key": api_key,
        "num": int(max(1, min(10, k))),
    }

    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    organic = data.get("organic_results") or []

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    results: List[Dict[str, Any]] = []
    for item in organic:
        link = item.get("link")
        title = item.get("title")
        snippet = item.get("snippet") or item.get("snippet_highlighted_words")
        if isinstance(snippet, list):
            snippet = " ".join(str(x) for x in snippet)

        if not link or not title:
            continue

        norm_url = _canonical_url(link)
        norm_title = _canonical_title(title)
        if norm_url in seen_urls or norm_title in seen_titles:
            continue
        seen_urls.add(norm_url)
        seen_titles.add(norm_title)

        results.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet or "",
                "source": source,
            }
        )

        if len(results) >= int(k):
            break

    return json.dumps({"query": query, "source": source, "results": results})


def web_build_playlist_impl(goal: str, k: int = 12, source: str = "spotify") -> str:
    queries = [
        f"{goal} playlist",
        f"{goal} songs",
        f"best {goal} songs",
    ]

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    items: List[Dict[str, Any]] = []
    for q in queries:
        payload = json.loads(web_search_songs_impl(q, k=min(10, k), source=source))
        for r in payload.get("results") or []:
            link = str(r.get("link") or "")
            title = str(r.get("title") or "")
            if not link or not title:
                continue
            norm_url = _canonical_url(link)
            norm_title = _canonical_title(title)
            if norm_url in seen_urls or norm_title in seen_titles:
                continue
            seen_urls.add(norm_url)
            seen_titles.add(norm_title)
            items.append(r)
            if len(items) >= k:
                break
        if len(items) >= k:
            break

    return json.dumps({"goal": goal, "source": source, "items": items})


@function_tool
def web_search_songs(query: str, k: int = 8, source: str = "spotify") -> str:
    """Search the web for real songs and return links + snippets.

    Args:
        query: Natural-language music query.
        k: Max results to return.
        source: One of: spotify, apple, genius, youtube, general.

    Returns:
        JSON string containing a list of results with title, link, snippet, and source.
    """

    return web_search_songs_impl(query=query, k=k, source=source)


@function_tool
def web_build_playlist(goal: str, k: int = 12, source: str = "spotify") -> str:
    """Build a playlist of real songs from the web.

    Args:
        goal: Natural-language playlist goal.
        k: Number of songs/links to return.
        source: One of: spotify, apple, genius, youtube, general.

    Returns:
        JSON string with playlist items (title/link/snippet).
    """

    return web_build_playlist_impl(goal=goal, k=k, source=source)
