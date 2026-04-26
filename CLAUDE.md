# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Music Recommender Simulation — an educational project demonstrating how recommendation systems work through three layered approaches: (1) scoring-based ranking, (2) RAG-enhanced semantic search, and (3) agent-based playlist building. The catalog is intentionally small (18 songs in `data/songs.csv`) to keep results explainable.

## Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env  # Fill in OPENAI_API_KEY and SERPAPI_API_KEY
```

### Run
```bash
# Primary UI (Streamlit web app)
streamlit run streamlit_app.py

# CLI modes
python -m src.main                                        # interactive scoring
python -m src.main search "sad indie breakup songs" --k 5
python -m src.main playlist "45 min study focus" --minutes 45 --unique-artist

# FastAPI server (docs at http://127.0.0.1:8000/docs)
uvicorn src.api:app --reload

# OmniAgents chat interface
omniagents run -c omniagent/agent.yml --mode web
omniagents run -c omniagent/agent.yml --mode server --port 9494
```

### Test
```bash
pytest                          # run all tests
pytest tests/test_recommender.py -k "test_recommend"   # run single test
```

## Architecture

### Three Recommendation Modes

**1. Scoring-based** (`src/recommender.py`)
- `Song` and `UserProfile` dataclasses; `Recommender` class wraps `score_song()` and `recommend_songs()`
- Scoring: genre match (+1.0), mood match (+1.5), energy proximity (+0.0–+4.0). **Energy dominates** — see model_card.md for bias implications.
- `acousticness` field exists in the data but is not scored

**2. RAG Search** (`src/rag/`)
- Pipeline: query expansion → `InMemoryVectorStore` (cosine similarity) → user profile reranking → explanation
- Embeddings via OpenAI API with a hash-based fallback when no API key
- `src/rag/rag_recommender.py`: `RagSearchEngine` is the main class
- `src/rag/song_docs.py`: converts `Song` dicts into searchable text documents

**3. Agent-based Playlist Building** (`src/agent/playlist_agent.py`)
- `PlaylistBuilderAgent` decomposes a natural language goal into subqueries, aggregates candidates from RAG, ranks them, and assembles a playlist with optional unique-artist constraint

### Dependency Injection

`src/system.py` uses `@lru_cache(maxsize=1)` to provide singletons: `get_vector_store()`, `get_search_engine()`, `get_playlist_agent()`. Import from here to avoid duplicate initialization.

### OmniAgents Integration (`omniagent/`)

- `agent.yml`: agent config (model, tools, temperature)
- `tools/music_tools.py`: local tools wrapping RAG + playlist agent
- `tools/web_song_tools.py`: web tools using SerpAPI for real Spotify/Apple Music links

### Entry Points

| Interface | File | Notes |
|-----------|------|-------|
| Streamlit UI | `streamlit_app.py` | Supports both local and remote agent modes |
| CLI | `src/main.py` | `search`, `playlist` subcommands |
| REST API | `src/api.py` | `/health`, `/search`, `/playlist` |
| Chat Agent | `omniagent/agent.yml` | OmniAgents-driven with web search |

`docs/recommender_flow.mmd` contains a Mermaid diagram of the recommendation pipeline.

### Data Flow
```
User input → [Streamlit / CLI / API / Chat]
                     ↓
              src/system.py (singletons)
              ↙            ↘
   RagSearchEngine    PlaylistBuilderAgent
   (embeddings +       (subqueries → RAG
   vector search)       → ranking → assembly)
```

## Key Known Limitations

From `model_card.md`:
- Energy score range (0–4.0) dominates genre/mood (+1.0/+1.5), leading users into narrow energy bands
- Acousticness feature in data is unused in scoring
- 18-song catalog limits diversity; some genres unrepresented
