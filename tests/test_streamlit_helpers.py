"""
Tests for helper functions in streamlit_app.py

Covers: _spotify_app_uri, _platform_links_html, try_parse_json,
        summarize_tool_payload.
No Streamlit runtime is needed — these are pure functions.
"""
from __future__ import annotations

import json

import pytest

from streamlit_app import (
    _platform_links_html,
    _spotify_app_uri,
    summarize_tool_payload,
    try_parse_json,
)


# ── _spotify_app_uri ──────────────────────────────────────────────────────────

class TestSpotifyAppUri:
    def test_track_url_produces_track_uri(self):
        url = "https://open.spotify.com/track/ABC123"
        result = _spotify_app_uri(url, "fallback+query")
        assert result == "spotify:track:ABC123"

    def test_album_url_produces_album_uri(self):
        url = "https://open.spotify.com/album/XYZ789"
        result = _spotify_app_uri(url, "fallback")
        assert result == "spotify:album:XYZ789"

    def test_playlist_url_produces_playlist_uri(self):
        url = "https://open.spotify.com/playlist/PL001"
        result = _spotify_app_uri(url, "fallback")
        assert result == "spotify:playlist:PL001"

    def test_artist_url_produces_artist_uri(self):
        url = "https://open.spotify.com/artist/AR999"
        result = _spotify_app_uri(url, "fallback")
        assert result == "spotify:artist:AR999"

    def test_tracking_params_stripped_before_id_extraction(self):
        url = "https://open.spotify.com/track/ABC123?si=tracking_token"
        result = _spotify_app_uri(url, "fallback")
        assert result == "spotify:track:ABC123"

    def test_non_spotify_url_falls_back_to_search_uri(self):
        url = "https://music.apple.com/us/album/xyz"
        result = _spotify_app_uri(url, "some+query")
        assert result == "spotify:search:some+query"

    def test_empty_url_falls_back_to_search_uri(self):
        result = _spotify_app_uri("", "sad+songs")
        assert result == "spotify:search:sad+songs"

    def test_unknown_spotify_path_type_falls_back(self):
        # "show" and "episode" are valid; anything else should fall back
        url = "https://open.spotify.com/unknowntype/ABC123"
        result = _spotify_app_uri(url, "fallback")
        assert result == "spotify:search:fallback"


# ── _platform_links_html ──────────────────────────────────────────────────────

class TestPlatformLinksHtml:
    def test_contains_spotify_button(self):
        html = _platform_links_html("Test Song", "")
        assert "btn-spotify" in html
        assert "Spotify" in html

    def test_contains_apple_music_button(self):
        html = _platform_links_html("Test Song", "")
        assert "btn-apple" in html
        assert "Apple Music" in html

    def test_contains_youtube_button(self):
        html = _platform_links_html("Test Song", "")
        assert "btn-youtube" in html
        assert "YouTube" in html

    def test_spotify_direct_url_used_when_primary_is_spotify(self):
        url = "https://open.spotify.com/track/ABC"
        html = _platform_links_html("Song", url)
        assert "open.spotify.com/track/ABC" in html

    def test_apple_direct_url_used_when_primary_is_apple(self):
        url = "https://music.apple.com/us/album/xyz"
        html = _platform_links_html("Song", url)
        assert "music.apple.com/us/album/xyz" in html

    def test_youtube_direct_url_used_when_primary_is_youtube(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        html = _platform_links_html("Song", url)
        assert "youtube.com/watch" in html

    def test_search_fallback_used_when_no_platform_url(self):
        html = _platform_links_html("Never Gonna Give You Up", "")
        assert "spotify.com/search" in html
        assert "apple.com" in html or "music.apple.com" in html
        assert "youtube.com" in html

    def test_all_buttons_have_href(self):
        html = _platform_links_html("Test Song", "")
        assert html.count('href="') >= 3


# ── try_parse_json ────────────────────────────────────────────────────────────

class TestTryParseJson:
    def test_valid_json_object_parsed(self):
        result = try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_list_parsed(self):
        result = try_parse_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_invalid_json_returns_none(self):
        assert try_parse_json("not json at all") is None

    def test_empty_string_returns_none(self):
        assert try_parse_json("") is None

    def test_whitespace_only_returns_none(self):
        assert try_parse_json("   ") is None

    def test_non_string_returns_none(self):
        assert try_parse_json({"already": "a dict"}) is None
        assert try_parse_json(None) is None
        assert try_parse_json(42) is None

    def test_json_with_leading_whitespace_parsed(self):
        result = try_parse_json('  {"key": 1}  ')
        assert result == {"key": 1}


# ── summarize_tool_payload ────────────────────────────────────────────────────

class TestSummarizeToolPayload:
    def _web_call(self, tool="web_search_songs", query="test", results=None):
        return {
            "type": "tool_output",
            "tool": tool,
            "output": json.dumps({
                "query": query,
                "source": "spotify",
                "results": results or [],
            }),
        }

    def _playlist_call(self, goal="study music", items=None):
        return {
            "type": "tool_output",
            "tool": "web_build_playlist",
            "output": json.dumps({
                "goal": goal,
                "source": "spotify",
                "items": items or [],
            }),
        }

    def test_web_search_output_classified_as_web(self):
        payload = summarize_tool_payload([self._web_call()])
        assert payload["kind"] == "web"

    def test_playlist_output_classified_as_web_playlist(self):
        payload = summarize_tool_payload([self._playlist_call()])
        assert payload["kind"] == "web_playlist"

    def test_web_payload_contains_data(self):
        payload = summarize_tool_payload([self._web_call(query="sad songs")])
        assert payload["data"]["query"] == "sad songs"

    def test_playlist_payload_contains_goal(self):
        payload = summarize_tool_payload([self._playlist_call(goal="workout mix")])
        assert payload["data"]["goal"] == "workout mix"

    def test_no_tool_output_returns_raw_kind(self):
        calls = [{"type": "tool_called", "tool": "web_search_songs", "arguments": "{}"}]
        payload = summarize_tool_payload(calls)
        assert payload["kind"] == "raw"

    def test_uses_last_tool_output_when_multiple_calls(self):
        calls = [
            self._web_call(query="first"),
            self._playlist_call(goal="second"),
        ]
        # reversed() picks the last one first — playlist call
        payload = summarize_tool_payload(calls)
        assert payload["kind"] == "web_playlist"

    def test_empty_tool_calls_returns_raw(self):
        payload = summarize_tool_payload([])
        assert payload["kind"] == "raw"

    def test_tool_name_stored_in_payload(self):
        payload = summarize_tool_payload([self._web_call(tool="web_search_songs")])
        assert payload["tool"] == "web_search_songs"
