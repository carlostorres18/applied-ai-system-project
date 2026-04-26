"""
Tests for omniagent/tools/web_song_tools.py

Covers: URL/title normalization and deduplication logic.
No real HTTP calls are made — requests.get is fully mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from omniagent.tools.web_song_tools import (
    _canonical_title,
    _canonical_url,
    web_build_playlist_impl,
    web_search_songs_impl,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_serpapi_response(results):
    """Build a mock requests.Response that returns the given organic_results."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"organic_results": results}
    return mock_resp


def organic_item(title, link, snippet="A snippet"):
    return {"title": title, "link": link, "snippet": snippet}


FAKE_KEY = "test-serpapi-key"


# ── _canonical_url ────────────────────────────────────────────────────────────

class TestCanonicalUrl:
    def test_strips_query_params(self):
        url = "https://open.spotify.com/track/abc123?si=xyz&utm_source=foo"
        assert _canonical_url(url) == "https://open.spotify.com/track/abc123"

    def test_strips_fragment(self):
        url = "https://open.spotify.com/track/abc#section"
        assert _canonical_url(url) == "https://open.spotify.com/track/abc"

    def test_strips_trailing_slash(self):
        url = "https://open.spotify.com/track/abc/"
        assert _canonical_url(url) == "https://open.spotify.com/track/abc"

    def test_leaves_clean_url_unchanged(self):
        url = "https://open.spotify.com/track/abc123"
        assert _canonical_url(url) == url

    def test_handles_invalid_url_gracefully(self):
        # Should return the original string, not raise
        result = _canonical_url("not a url at all ://??")
        assert isinstance(result, str)

    def test_empty_string_returns_empty(self):
        assert _canonical_url("") == ""


# ── _canonical_title ──────────────────────────────────────────────────────────

class TestCanonicalTitle:
    def test_lowercases_title(self):
        assert _canonical_title("BLINDING LIGHTS") == "blinding lights"

    def test_strips_whitespace(self):
        assert _canonical_title("  Hello World  ") == "hello world"

    def test_empty_string_returns_empty(self):
        assert _canonical_title("") == ""

    def test_mixed_case_and_spaces(self):
        assert _canonical_title("  Song Title  ") == "song title"


# ── web_search_songs_impl (deduplication) ─────────────────────────────────────

class TestWebSearchSongsImpl:
    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_returns_json_with_results_key(self, mock_get, _):
        mock_get.return_value = make_serpapi_response([
            organic_item("Song A", "https://open.spotify.com/track/aaa"),
        ])
        payload = json.loads(web_search_songs_impl("happy songs", k=5))
        assert "results" in payload
        assert "query" in payload

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_deduplicates_same_url_with_different_tracking_params(self, mock_get, _):
        mock_get.return_value = make_serpapi_response([
            organic_item("Song A", "https://open.spotify.com/track/abc?si=111"),
            organic_item("Song A", "https://open.spotify.com/track/abc?si=222"),
        ])
        payload = json.loads(web_search_songs_impl("songs", k=5))
        assert len(payload["results"]) == 1

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_deduplicates_same_title_different_url(self, mock_get, _):
        mock_get.return_value = make_serpapi_response([
            organic_item("Blinding Lights", "https://open.spotify.com/track/aaa"),
            organic_item("Blinding Lights", "https://open.spotify.com/track/bbb"),
        ])
        payload = json.loads(web_search_songs_impl("songs", k=5))
        assert len(payload["results"]) == 1

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_respects_k_limit(self, mock_get, _):
        items = [organic_item(f"Song {i}", f"https://spotify.com/track/{i}") for i in range(10)]
        mock_get.return_value = make_serpapi_response(items)
        payload = json.loads(web_search_songs_impl("songs", k=3))
        assert len(payload["results"]) <= 3

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_skips_items_missing_title_or_link(self, mock_get, _):
        mock_get.return_value = make_serpapi_response([
            {"title": None, "link": "https://spotify.com/track/aaa", "snippet": ""},
            {"title": "Good Song", "link": None, "snippet": ""},
            organic_item("Valid Song", "https://spotify.com/track/bbb"),
        ])
        payload = json.loads(web_search_songs_impl("songs", k=5))
        assert len(payload["results"]) == 1
        assert payload["results"][0]["title"] == "Valid Song"

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_source_field_included_in_results(self, mock_get, _):
        mock_get.return_value = make_serpapi_response([
            organic_item("Song A", "https://open.spotify.com/track/aaa"),
        ])
        payload = json.loads(web_search_songs_impl("songs", k=5, source="spotify"))
        assert payload["results"][0]["source"] == "spotify"

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.requests.get")
    def test_empty_results_from_serpapi_returns_empty_list(self, mock_get, _):
        mock_get.return_value = make_serpapi_response([])
        payload = json.loads(web_search_songs_impl("obscure query xyz", k=5))
        assert payload["results"] == []

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value="")
    def test_missing_api_key_raises_runtime_error(self, _):
        with pytest.raises(RuntimeError, match="SERPAPI"):
            web_search_songs_impl("songs")


# ── web_build_playlist_impl (cross-query dedup) ───────────────────────────────

class TestWebBuildPlaylistImpl:
    def _mock_search(self, results_per_query):
        """Return a side_effect list for web_search_songs_impl calls."""
        return [
            json.dumps({"query": "q", "source": "spotify", "results": r})
            for r in results_per_query
        ]

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.web_search_songs_impl")
    def test_returns_json_with_items_and_goal(self, mock_search, _):
        mock_search.return_value = json.dumps({
            "query": "study", "source": "spotify",
            "results": [organic_item("Song A", "https://spotify.com/track/a")],
        })
        payload = json.loads(web_build_playlist_impl("study music", k=5))
        assert "items" in payload
        assert "goal" in payload
        assert payload["goal"] == "study music"

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.web_search_songs_impl")
    def test_deduplicates_same_song_across_queries(self, mock_search, _):
        # All three queries return the same song
        same = [organic_item("Song A", "https://spotify.com/track/aaa")]
        mock_search.return_value = json.dumps({
            "query": "q", "source": "spotify", "results": same,
        })
        payload = json.loads(web_build_playlist_impl("study", k=10))
        assert len(payload["items"]) == 1

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.web_search_songs_impl")
    def test_aggregates_unique_songs_across_queries(self, mock_search, _):
        calls = [
            json.dumps({"query": "q", "source": "spotify", "results": [
                organic_item("Song A", "https://spotify.com/track/a"),
            ]}),
            json.dumps({"query": "q", "source": "spotify", "results": [
                organic_item("Song B", "https://spotify.com/track/b"),
            ]}),
            json.dumps({"query": "q", "source": "spotify", "results": [
                organic_item("Song C", "https://spotify.com/track/c"),
            ]}),
        ]
        mock_search.side_effect = calls
        payload = json.loads(web_build_playlist_impl("study", k=10))
        assert len(payload["items"]) == 3

    @patch("omniagent.tools.web_song_tools._get_serpapi_key", return_value=FAKE_KEY)
    @patch("omniagent.tools.web_song_tools.web_search_songs_impl")
    def test_respects_k_limit_across_queries(self, mock_search, _):
        many = [organic_item(f"Song {i}", f"https://spotify.com/track/{i}") for i in range(10)]
        mock_search.return_value = json.dumps({
            "query": "q", "source": "spotify", "results": many,
        })
        payload = json.loads(web_build_playlist_impl("party", k=4))
        assert len(payload["items"]) <= 4
