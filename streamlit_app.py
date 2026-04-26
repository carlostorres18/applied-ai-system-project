from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


@dataclass
class AgentRunResult:
    assistant_text: str
    tool_calls: List[Dict[str, Any]]


# ── Styles ────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.hero-icon { font-size: 52px; line-height: 1; }
.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 5px 0;
    letter-spacing: -0.5px;
}
.hero-sub { font-size: 14px; color: rgba(255,255,255,0.55); margin: 0; }

/* Suggestion chips */
.suggestions { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 6px; }
.chip {
    display: inline-block;
    padding: 8px 18px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 999px;
    font-size: 13px;
    color: #a5b4fc;
}

/* Section label badges */
.section-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #818cf8;
}
.result-count {
    font-size: 12px;
    opacity: 0.5;
    margin-left: 8px;
}

/* Playlist header */
.playlist-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin: 10px 0 14px;
}
.playlist-title { font-size: 18px; font-weight: 700; margin: 0; }
.playlist-count {
    font-size: 12px;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border-radius: 999px;
    padding: 2px 10px;
    font-weight: 600;
}

/* Song / result cards */
.song-card {
    position: relative;
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 14px 16px 14px 16px;
    margin-bottom: 10px;
    background: rgba(99,102,241,0.04);
    transition: border-color 0.2s ease, background 0.2s ease;
}
.song-card:hover {
    border-color: rgba(99,102,241,0.45);
    background: rgba(99,102,241,0.09);
}
.track-badge {
    position: absolute;
    top: 13px;
    right: 15px;
    font-size: 11px;
    font-weight: 700;
    color: rgba(99,102,241,0.45);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.song-title {
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 5px;
    padding-right: 56px;
}
.song-snippet {
    font-size: 13px;
    opacity: 0.68;
    margin-bottom: 12px;
    line-height: 1.55;
}

/* Platform buttons */
.platform-links { display: flex; gap: 8px; flex-wrap: wrap; }
.platform-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none !important;
    transition: opacity 0.15s ease, transform 0.1s ease;
}
.platform-btn:hover { opacity: 0.82; transform: translateY(-1px); }
.btn-spotify      { background: #1DB954; color: #fff !important; }
.btn-spotify-web  { background: transparent; color: #1DB954 !important; border: 1px solid #1DB954; padding: 5px 9px; }
.btn-apple        { background: #FC3C44; color: #fff !important; }
.btn-youtube      { background: #FF0000; color: #fff !important; }
</style>
"""

SUGGESTIONS = [
    "Chill lo-fi study beats",
    "Upbeat songs for working out",
    "Build me a 30 min focus playlist",
    "Sad indie songs for a rainy day",
]


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Music Agent", page_icon="🎵", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero
    st.markdown(
        """
<div class="hero">
  <div class="hero-icon">🎵</div>
  <div>
    <p class="hero-title">Music Agent</p>
    <p class="hero-sub">
      AI-powered recommendations &amp; playlist builder &nbsp;·&nbsp;
      Spotify &nbsp;·&nbsp; Apple Music &nbsp;·&nbsp; YouTube
    </p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    config_path = Path("omniagent/agent.yml").resolve()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        model_override = st.selectbox(
            "Model",
            options=["gpt-5.2", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
            index=0,
        )

        st.divider()
        st.markdown("**💡 Try asking...**")
        st.markdown(
            "- *Real songs from Spotify*\n"
            "- *45 min study playlist*\n"
            "- *Sad breakup songs*\n"
            "- *High energy workout mix*"
        )

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["chat"] = []
            st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────────
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    # Welcome / suggestion chips shown only when chat is empty
    if not st.session_state["chat"]:
        st.markdown("**What are you in the mood for?**")
        chips_html = "".join(f'<span class="chip">{s}</span>' for s in SUGGESTIONS)
        st.markdown(
            f'<div class="suggestions">{chips_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:12px;opacity:0.4;margin-top:4px;'>"
            "Type one of the suggestions above or ask anything in the chat box below.</p>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            payload = msg.get("tool_payload")
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif payload and payload.get("kind") in ("web", "web_playlist"):
                render_tool_payload(payload)
            else:
                st.markdown("_No web results found. Try asking for songs from Spotify or a playlist._")

    # ── Chat input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask for songs, a playlist, or real web results...")
    if not prompt:
        return

    st.session_state["chat"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding music for you..."):
            result = run_omniagent_local(
                config_path=str(config_path),
                user_text=prompt,
                model_name=str(model_override),
            )

        tool_payload = summarize_tool_payload(result.tool_calls) if result.tool_calls else None

        if tool_payload and tool_payload.get("kind") in ("web", "web_playlist"):
            render_tool_payload(tool_payload)
        else:
            st.markdown("_No web results found. Try asking for songs from Spotify or a playlist._")

    st.session_state["chat"].append(
        {
            "role": "assistant",
            "content": result.assistant_text or "(No assistant text returned.)",
            "tool_payload": tool_payload,
        }
    )


# ── Rendering helpers ─────────────────────────────────────────────────────────
def summarize_tool_payload(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    for call in reversed(tool_calls):
        if call.get("type") != "tool_output":
            continue
        name = call.get("tool")
        raw = call.get("output")
        parsed = try_parse_json(raw)
        if isinstance(parsed, dict) and "results" in parsed and "query" in parsed:
            return {"kind": "web", "tool": name, "data": parsed}
        if isinstance(parsed, dict) and "items" in parsed and "goal" in parsed:
            return {"kind": "web_playlist", "tool": name, "data": parsed}
    return {"kind": "raw", "tool_calls": tool_calls}


def render_tool_payload(payload: Dict[str, Any]) -> None:
    kind = payload.get("kind")

    if kind == "web":
        data = payload.get("data") or {}
        results = data.get("results") or []
        n = len(results)
        st.markdown(
            f'<div style="margin:10px 0 14px;">'
            f'<span class="section-label">🌐 Web Search</span>'
            f'<span class="result-count">{n} result{"s" if n != 1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for i, r in enumerate(results, 1):
            links_html = _platform_links_html(
                title=str(r.get("title") or ""),
                primary_url=str(r.get("link") or ""),
            )
            st.markdown(
                f"""
<div class="song-card">
  <span class="track-badge">#{i}</span>
  <div class="song-title">{r.get('title', '')}</div>
  <div class="song-snippet">{r.get('snippet', '')}</div>
  {links_html}
</div>""",
                unsafe_allow_html=True,
            )
        return

    if kind == "web_playlist":
        data = payload.get("data") or {}
        items = data.get("items") or []
        goal = data.get("goal", "")
        n = len(items)
        st.markdown(
            f'<div style="margin:10px 0 4px;">'
            f'<span class="section-label">🎧 Playlist Builder</span>'
            f'</div>'
            f'<div class="playlist-header">'
            f'<span class="playlist-title">{goal}</span>'
            f'<span class="playlist-count">{n} track{"s" if n != 1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for i, item in enumerate(items, 1):
            links_html = _platform_links_html(
                title=str(item.get("title") or ""),
                primary_url=str(item.get("link") or ""),
            )
            st.markdown(
                f"""
<div class="song-card">
  <span class="track-badge">Track {i}</span>
  <div class="song-title">{item.get('title', '')}</div>
  <div class="song-snippet">{item.get('snippet', '')}</div>
  {links_html}
</div>""",
                unsafe_allow_html=True,
            )
        return


def _spotify_app_uri(primary_url: str, encoded_query: str) -> str:
    """Return a spotify: deep-link URI.

    For a known Spotify HTTPS URL (track/album/playlist/artist) extract the ID
    and return e.g. ``spotify:track:ABC123``.  For everything else return a
    search URI so the app opens to matching results.
    """
    from urllib.parse import urlparse

    primary_lower = (primary_url or "").lower()
    if "open.spotify.com" in primary_lower:
        try:
            parts = [p for p in urlparse(primary_url).path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {
                "track", "album", "playlist", "artist", "episode", "show"
            }:
                return f"spotify:{parts[0]}:{parts[1]}"
        except Exception:
            pass
    return f"spotify:search:{encoded_query}"


def _spotify_deep_link_html(app_uri: str, web_url: str) -> str:
    """Return Spotify buttons that open the app via direct href deep-link.

    Browsers pass clicked href custom-scheme links (spotify:) to the OS even
    from inside iframes, unlike JS window.location navigations which are
    blocked by iframe sandbox rules.  A small 'web' text link sits alongside
    as a fallback for users without the app installed.
    """
    return (
        f'<a href="{app_uri}" class="platform-btn btn-spotify">&#9835; Spotify</a>'
        f'<a href="{web_url}" target="_blank" class="platform-btn btn-spotify-web">&#8599;</a>'
    )


def _platform_links_html(title: str, primary_url: str) -> str:
    q = quote_plus(title.strip())
    primary_lower = (primary_url or "").lower()

    # Spotify — app deep-link with HTTPS fallback
    spotify_web = (
        primary_url if "open.spotify.com" in primary_lower
        else f"https://open.spotify.com/search/{q}"
    )
    spotify_btn = _spotify_deep_link_html(
        app_uri=_spotify_app_uri(primary_url, q),
        web_url=spotify_web,
    )

    apple_url = (
        primary_url if "music.apple.com" in primary_lower
        else f"https://music.apple.com/us/search?term={q}"
    )
    youtube_url = (
        primary_url if ("youtube.com" in primary_lower or "youtu.be" in primary_lower)
        else f"https://www.youtube.com/results?search_query={q}"
    )

    return (
        f'<div class="platform-links">'
        f'{spotify_btn}'
        f'<a href="{apple_url}" target="_blank" class="platform-btn btn-apple">&#63743; Apple Music</a>'
        f'<a href="{youtube_url}" target="_blank" class="platform-btn btn-youtube">&#9654; YouTube</a>'
        f'</div>'
    )


def try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


# ── Agent runners ─────────────────────────────────────────────────────────────
def run_omniagent_local(config_path: str, user_text: str, model_name: str) -> AgentRunResult:
    from agents import ItemHelpers
    from omniagents.core.agents.factory import create_unified_agent
    from omniagents.core.config.loader import load_agent_spec_from_yaml

    async def _run() -> AgentRunResult:
        spec = load_agent_spec_from_yaml(config_path)
        spec.model_name = model_name
        agent = await create_unified_agent(spec)

        result = agent.run_streamed([{"role": "user", "content": user_text}])

        assistant_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        async for event in result.stream_events():
            name = getattr(event, "name", None)
            item = getattr(event, "item", None)

            if name == "message_output_created" and item is not None:
                raw = getattr(item, "raw_item", None)
                text = ItemHelpers.extract_last_text(raw) if raw is not None else None
                if text:
                    assistant_parts.append(str(text))

            if name == "tool_called" and item is not None:
                tool_calls.append(
                    {
                        "type": "tool_called",
                        "tool": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", None),
                    }
                )

            if name == "tool_output" and item is not None:
                tool_calls.append(
                    {
                        "type": "tool_output",
                        "tool": getattr(item, "name", None),
                        "output": getattr(item, "output", None),
                    }
                )

        assistant_text = "\n".join([p for p in assistant_parts if p.strip()]).strip()
        if not assistant_text:
            try:
                last = getattr(agent, "get_last_run_end", lambda: None)()
            except Exception:
                last = None
            if last and isinstance(last, dict) and last.get("error"):
                assistant_text = "(Agent error)\n\n" + json.dumps(last.get("error"), indent=2)

        return AgentRunResult(assistant_text=assistant_text, tool_calls=tool_calls)

    try:
        return asyncio.run(_run())
    except Exception as e:
        return AgentRunResult(assistant_text=f"(Agent exception)\n\n{e}", tool_calls=[])


def run_omniagent_remote(port: int, user_text: str) -> AgentRunResult:
    from agents import ItemHelpers
    from omniagents.rpc.agents.remote import RemoteAgent

    async def _run() -> AgentRunResult:
        ws_url = f"ws://127.0.0.1:{port}/ws"
        agent = st.session_state.get("_remote_agent")
        if not isinstance(agent, RemoteAgent):
            agent = RemoteAgent(ws_url)
            st.session_state["_remote_agent"] = agent

        result = agent.run_streamed([{"role": "user", "content": user_text}])

        assistant_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        async for event in result.stream_events():
            name = getattr(event, "name", None)
            item = getattr(event, "item", None)

            if name == "message_output_created" and item is not None:
                raw = getattr(item, "raw_item", None)
                text = ItemHelpers.extract_last_text(raw) if raw is not None else None
                if text:
                    assistant_parts.append(str(text))

            if name == "tool_called" and item is not None:
                tool_calls.append(
                    {
                        "type": "tool_called",
                        "tool": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", None),
                    }
                )

            if name == "tool_output" and item is not None:
                tool_calls.append(
                    {
                        "type": "tool_output",
                        "tool": getattr(item, "name", None),
                        "output": getattr(item, "output", None),
                    }
                )

        assistant_text = "\n".join([p for p in assistant_parts if p.strip()]).strip()
        if not assistant_text:
            last = agent.get_last_run_end() if hasattr(agent, "get_last_run_end") else None
            if last and isinstance(last, dict) and last.get("error"):
                assistant_text = "(Agent error)\n\n" + json.dumps(last.get("error"), indent=2)

        return AgentRunResult(assistant_text=assistant_text, tool_calls=tool_calls)

    try:
        return asyncio.run(_run())
    except Exception as e:
        return AgentRunResult(assistant_text=f"(Agent exception)\n\n{e}", tool_calls=[])


def write_agent_override_config(base_config_path: Path, model_name: str) -> Path:
    import yaml

    with base_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = model_name

    out_dir = Path(".omniagent_runtime")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agent.override.yml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    main()
