# Music Recommender Agent

An AI-powered music recommendation system that combines a scoring-based recommender, RAG semantic search, and an agent-driven chat interface to suggest real songs with direct links to Spotify, Apple Music, and YouTube.

---

## What It Does

The project layers three recommendation approaches on top of each other:

| Layer | What it does |
|-------|-------------|
| **Scoring engine** | Scores each song in an 18-song local catalog against a user taste profile (genre, mood, energy) and ranks them |
| **RAG search** | Converts natural-language queries into vector embeddings, searches the catalog semantically, and re-ranks results using user profile affinity |
| **Web agent** | Uses an OmniAgents-powered chat interface to search the real web (via SerpAPI) and return actual Spotify/Apple Music/YouTube links |

The primary interface is a Streamlit chat app. You type what you want to hear and the agent returns real songs with platform buttons to open them directly.

> **Important:** The Streamlit chat app uses only the web agent layer. It does **not** use `data/songs.csv`, the local scoring engine, or the `UserProfile`. Results come entirely from live web searches via SerpAPI and are not personalized to any stored taste profile. The local catalog and scoring logic are only exercised through the CLI and FastAPI server (see [Other Ways to Run](#other-ways-to-run)).

---

## How the Scoring Works

Each song in the local catalog is scored against a `UserProfile`:

- **Genre match** — +1.0 if the song's genre matches the user's favorite
- **Mood match** — +1.5 if the song's mood matches
- **Energy proximity** — up to +4.0 based on how close the song's energy is to the user's target

Songs are ranked by total score and the top-k are returned with a plain-language explanation.

> Energy has the highest possible value (4.0) so it dominates the ranking. A song can rank high even if the genre or mood doesn't match, as long as its energy is close to the target.
---

## Project Structure

```
streamlit_app.py          # Main web UI (Streamlit chat app)
src/
  recommender.py          # Song, UserProfile, Recommender — core scoring logic
  rag/                    # RAG pipeline (embeddings, vector store, reranking)
  agent/playlist_agent.py # PlaylistBuilderAgent (goal → subqueries → ranked playlist)
  api.py                  # FastAPI REST endpoints (/search, /playlist)
  main.py                 # CLI entry point
omniagent/
  agent.yml               # OmniAgents config (model, tools)
  instructions.md         # Agent system prompt
  tools/
    web_song_tools.py     # web_search_songs and web_build_playlist (SerpAPI)
    music_tools.py        # Local catalog tools (search_songs, build_playlist)
data/songs.csv            # 18-song catalog
tests/                    # Unit tests
```

---

## Getting Started

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd applied-ai-system-project
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

```
OPENAI_BASE_URL=https://your-openai-base-url/v1
OPENAI_API_KEY=sk-...
SERPAPI_API_KEY=your-serpapi-key-here
```

- **OPENAI_API_KEY** — used for generating text embeddings in the RAG layer. A hash-based fallback is used if this is missing.
- **SERPAPI_API_KEY** — required for the web search tools (`web_search_songs`, `web_build_playlist`). Without this the agent cannot fetch real song links.

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

This opens the chat UI in your browser. Type a request and the agent returns real songs with Spotify, Apple Music, and YouTube links.

---

## Using the App

**Search for songs** — describe a vibe, mood, or genre:
> "Sad indie songs for a rainy day"
> "Upbeat pop songs for working out"

**Build a playlist** — mention a duration or activity:
> "Build me a 30 min focus playlist"
> "45 minute workout mix"

The **Spotify** button attempts to open the track directly in the Spotify desktop app if you have it installed, and falls back to the Spotify website. The **Apple Music** and **YouTube** buttons always open in the browser.

### Sidebar controls

| Control | What it does |
|---------|-------------|
| **Model** | Swap the AI model the agent uses (gpt-5.2, gpt-4o, gpt-4o-mini, gpt-4.1-mini) |
| **Try asking...** | Example prompts to get started |
| **Clear chat** | Wipes the conversation and returns to the welcome screen |

---

## Other Ways to Run

### CLI

```bash
# Score-based recommendations (local catalog)
python -m src.main

# RAG natural-language search
python -m src.main search "late night coding vibes" --k 5

# Playlist builder agent
python -m src.main playlist "45 minutes study focus" --minutes 45 --unique-artist
```

### FastAPI server

```bash
uvicorn src.api:app --reload
# Docs at http://127.0.0.1:8000/docs
```

### Tests

```bash
pytest
pytest tests/test_recommender.py -k "test_recommend"   # single test
```

---

## Limitations

- **Energy dominates scoring.** The energy proximity component can score up to +4.0 vs. +1.0 for genre and +1.5 for mood. A song with the wrong genre but matching energy will often rank above a better mood/genre match.
- **Acousticness is unused.** The `acousticness` field exists in the catalog and user profile but is never factored into the score.
- **Small local catalog.** The scoring and RAG layers only have 18 songs. Some genres (classical, metal) are underrepresented or missing entirely.
- **No lyrics or language understanding.** All recommendations are based on metadata features, not what a song actually sounds like or says.
- **Web results depend on SerpAPI.** If the API key is missing or the quota is exceeded, the agent cannot return any results.
- **No user history.** Every conversation starts fresh — there is no memory of past preferences across sessions.
- **`songs.csv` and `UserProfile` are not used by the chat app.** The Streamlit interface bypasses the local catalog and scoring engine entirely. The 18-song catalog, genre/mood/energy scoring, and `UserProfile` personalization only apply when running via the CLI or FastAPI server.


---

## How the Web Agent Works

When you send a message in the Streamlit chat:

1. The OmniAgents framework receives your text and decides which tool to call
2. For song searches it calls `web_search_songs`, which queries Google via SerpAPI with a Spotify site filter and returns deduplicated results
3. For playlist requests it calls `web_build_playlist`, which runs three query variations, aggregates the results, and deduplicates by URL and title
4. The results are rendered as cards with numbered tracks and platform buttons

The agent only calls one of the two web tools per request — it does not use the local catalog in the chat interface.
