# Model Card — Music Recommender Agent

**Project:** Music Recommender Agent  
**Author:** Carlos Torres  
**Date:** April 2026  
**Base System:** OmniAgents + OpenAI (GPT-4o-mini) + SerpAPI + Streamlit

---

## What This System Does

This system is an AI-powered music discovery chatbot. A user types a natural language query ("sad breakup songs", "45-minute study playlist"), and the agent uses LLM-driven tool routing to call a SerpAPI web search, retrieve real Spotify/Apple Music/YouTube results, and return them as a formatted card-based UI with direct platform links.

The underlying architecture layers three recommendation approaches:
1. **Scoring Engine** — rule-based: genre (+1.0), mood (+1.5), energy proximity (0–4.0)
2. **RAG Search** — OpenAI embeddings + cosine similarity over an 18-song local catalog
3. **Web Agent** — live SerpAPI search for real-world results (the primary chat interface)

---

## AI Collaboration Reflections

### One Suggestion That Was Genuinely Helpful

**Suggestion:** Using an `autouse=True` pytest fixture in `conftest.py` to patch `embed_texts` globally across all tests, so no test file needs to manually mock OpenAI calls.

**Why it helped:** The original test suite was calling OpenAI's embedding API on every test run, which made tests slow, flaky, and required a live API key in CI. The `autouse` fixture intercepts the embedding call at the module level and replaces it with a deterministic hash-based fallback. This made 148 tests run in under 2 seconds with zero network calls. It was a non-obvious Python testing pattern — using `unittest.mock.patch` inside a `yield` fixture — that significantly improved test reliability and portability.

### One Suggestion That Was Flawed

**Suggestion:** Opening the Spotify app via JavaScript (`window.location.href = 'spotify:track:ID'`) injected into a Streamlit component.

**Why it failed:** Streamlit renders components inside an `<iframe>`. Browsers apply a sandbox policy to iframes that blocks navigation to custom URI schemes (`spotify://`, `spotify:track:`) via JavaScript. The JS executed silently with no error, but the Spotify app never opened. The fix was to use a plain HTML anchor tag (`<a href="spotify:track:ABC123">`) rendered via `st.markdown(..., unsafe_allow_html=True)`. The browser handles `href=` URI scheme navigation at the top-level document context, bypassing the iframe sandbox restriction. The initial JS suggestion sounded correct but ignored how Streamlit's iframe architecture interacts with the browser security model.

---

## Biases and Limitations

### 1. Energy Score Dominance

The scoring engine uses the following weights:
- Genre match: **+1.0** (binary)
- Mood match: **+1.5** (binary)
- Energy proximity: **0.0 – +4.0** (continuous, range of 4 points)

Because energy spans a continuous 4-point range while genre and mood together cap out at 2.5, a song with a strong energy match will almost always outrank a song with a perfect genre+mood match but slightly mismatched energy. This creates a feedback loop: if a user is profiled as preferring high energy, they will be routed into high-energy songs even when they ask for something mellow. The chat interface (web agent) is not affected since it bypasses the scoring engine entirely and uses live web search — but the CLI and API endpoints exhibit this bias.

### 2. Acousticness Feature Unused

The `songs.csv` catalog includes an `acousticness` field for every song (range 0.0–1.0), but the `score_song()` function never reads it. A user who explicitly prefers acoustic music ("unplugged", "acoustic guitar") receives no scoring benefit from this signal. The field is indexed in RAG documents (via `build_song_doc`), so semantic search may partially surface acoustic songs through keyword matching, but the scoring layer ignores it entirely.

### 3. Catalog Size and Genre Underrepresentation

The local catalog contains exactly 18 songs. This means:
- Entire genres (jazz, classical, country, reggae, R&B, K-pop, etc.) are absent
- Users asking for genre-specific recommendations via the scoring/RAG layers will receive results from the closest available genre
- The web agent (SerpAPI) does not have this limitation since it searches the open web

### 4. No Explicit Diversity Enforcement in Web Results

The web agent deduplicated results by URL and title, but it does not enforce artist diversity across the final playlist. A SerpAPI search for "sad love songs" could return five tracks by the same artist if they dominate the results page. The local `PlaylistBuilderAgent` enforces unique-artist constraints; the web agent does not.

---

## Testing Results

### Test Suite Overview

| File | Tests | What It Covers |
|------|-------|----------------|
| `tests/test_recommender.py` | 27 | `score_song`, `recommend_songs`, `Recommender`, `load_songs` |
| `tests/test_rag.py` | 36 | Hash embedding, cosine similarity, vector store, query expansion, reranking, `RagSearchEngine` |
| `tests/test_playlist_agent.py` | 34 | Subquery generation, constraint filtering, assembly, ranking, title generation, `PlaylistBuilderAgent` |
| `tests/test_web_tools.py` | 26 | URL/title normalization, deduplication, SerpAPI mocking, `web_search_songs_impl`, `web_build_playlist_impl` |
| `tests/test_streamlit_helpers.py` | 25 | `_spotify_app_uri`, `_platform_links_html`, `try_parse_json`, `summarize_tool_payload` |
| **Total** | **148** | |

All 148 tests pass with zero live API calls. The `conftest.py` autouse fixture patches `embed_texts` globally so OpenAI is never contacted during testing.

### What Worked

- **Deduplication logic** is fully verified: URL-based dedup (stripping `?si=` tracking params), title-based dedup (case-insensitive), and cross-query dedup in playlist building all behave correctly under test.
- **Scoring weights** are verified to be numerically correct for all combinations of genre/mood match and energy proximity.
- **RAG pipeline** is verified end-to-end using hash-based embeddings: queries return results, filters apply correctly, reranking adjusts scores by user profile.
- **Playlist assembly** correctly enforces the `unique_artist` constraint by default and respects the `target_count` cap.

### What Didn't Work (and What Was Learned)

- **JS-based Spotify deep-link**: Failed silently inside Streamlit's iframe. Discovered that custom URI scheme navigation (`spotify:`, `mailto:`, etc.) must use `<a href=...>` rather than JS assignment to work reliably in embedded contexts.
- **Pytest module resolution**: Tests initially failed with `ModuleNotFoundError: No module named 'src'`. Resolved by adding `pytest.ini` with `pythonpath = .`. This is a common Python project setup issue when the test runner doesn't automatically include the project root on `sys.path`.
- **SerpAPI result variability**: Live SerpAPI calls occasionally return different result counts (ads, featured snippets mixed into organic results). The `k` parameter cap and deduplication handle this gracefully, but it means playlist length can vary between identical queries.

---

## Responsible AI Notes

- **No user data is stored.** All chat history is session-local in Streamlit's `st.session_state` and discarded when the browser tab closes.
- **No personalization profile is inferred.** The chat UI does not build or store a user profile across sessions; it only uses the live query.
- **SerpAPI results are real web links.** The system surfaces links it did not generate, so it inherits whatever biases, regional availability restrictions, or takedowns affect Spotify/Apple Music/YouTube search results.
- **The 18-song local catalog is intentionally small** to keep results explainable in an educational context, not to reflect what an actual production recommender would serve.

---

## What This Project Says About Me as an AI Engineer

Building this project showed me that AI engineering is less about knowing which model to call and more about understanding what happens between the model and the user. I ran into failures that had nothing to do with the AI itself — a browser sandbox blocking a URI scheme, a pytest runner not knowing where to find my modules, tracking parameters making duplicate songs look unique. Solving those edge cases required reading how systems actually behave, not just how they're documented. I also learned to think critically about what I build: the energy bias in the scoring engine, the unused acousticness field, the fact that a 148-test suite gives you confidence but not certainty. I leave this project knowing that responsible AI work means documenting what your system gets wrong just as clearly as what it gets right — and that the most valuable engineering skill is being honest about the gap between the two.

