"""
Tests for src/recommender.py

Covers: score_song(), recommend_songs(), load_songs(), Recommender class.
"""
from __future__ import annotations

import csv
import tempfile
import os

import pytest

from src.recommender import (
    Recommender,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

POP_HAPPY_HIGH = Song(
    id=1, title="Pop Banger", artist="Artist A",
    genre="pop", mood="happy", energy=0.9,
    tempo_bpm=130, valence=0.9, danceability=0.85, acousticness=0.1,
)
LOFI_CHILL_LOW = Song(
    id=2, title="Chill Loop", artist="Artist B",
    genre="lofi", mood="chill", energy=0.3,
    tempo_bpm=75, valence=0.5, danceability=0.4, acousticness=0.8,
)
ROCK_INTENSE_MID = Song(
    id=3, title="Rock Out", artist="Artist C",
    genre="rock", mood="intense", energy=0.7,
    tempo_bpm=140, valence=0.6, danceability=0.6, acousticness=0.05,
)

ALL_SONGS = [POP_HAPPY_HIGH, LOFI_CHILL_LOW, ROCK_INTENSE_MID]


def make_recommender(songs=None):
    return Recommender(songs or ALL_SONGS)


# ── score_song ────────────────────────────────────────────────────────────────

class TestScoreSong:
    def test_full_match_scores_higher_than_no_match(self):
        user = {"genre": "pop", "mood": "happy", "energy": 0.9}
        song_match = {"genre": "pop", "mood": "happy", "energy": 0.9}
        song_none  = {"genre": "jazz", "mood": "sad",  "energy": 0.1}

        score_match, _ = score_song(user, song_match)
        score_none,  _ = score_song(user, song_none)

        assert score_match > score_none

    def test_genre_match_adds_points(self):
        user = {"genre": "pop", "mood": "", "energy": 0.5}
        with_genre    = {"genre": "pop",  "mood": "", "energy": 0.5}
        without_genre = {"genre": "rock", "mood": "", "energy": 0.5}

        score_with,    _ = score_song(user, with_genre)
        score_without, _ = score_song(user, without_genre)

        assert score_with > score_without

    def test_mood_match_adds_points(self):
        user = {"genre": "", "mood": "happy", "energy": 0.5}
        with_mood    = {"genre": "", "mood": "happy", "energy": 0.5}
        without_mood = {"genre": "", "mood": "sad",   "energy": 0.5}

        score_with,    _ = score_song(user, with_mood)
        score_without, _ = score_song(user, without_mood)

        assert score_with > score_without

    def test_energy_proximity_perfect_match_gives_max_points(self):
        user = {"genre": "", "mood": "", "energy": 0.5}
        song = {"genre": "", "mood": "", "energy": 0.5}

        score, reasons = score_song(user, song)

        # Max energy points = 4.0; with perfect match delta=0 → 4.0 - 0 = 4.0
        assert score == pytest.approx(4.0, abs=0.01)
        assert any("energy" in r for r in reasons)

    def test_energy_proximity_far_miss_gives_zero_energy_points(self):
        user = {"genre": "", "mood": "", "energy": 0.0}
        song = {"genre": "", "mood": "", "energy": 1.0}

        score, _ = score_song(user, song)

        # delta=1.0 → 4.0 - (1.0 * 4.0) = 0.0
        assert score == pytest.approx(0.0, abs=0.01)

    def test_returns_non_empty_reasons_list(self):
        user = {"genre": "pop", "mood": "happy", "energy": 0.8}
        song = {"genre": "pop", "mood": "happy", "energy": 0.8}

        _, reasons = score_song(user, song)

        assert isinstance(reasons, list)
        assert len(reasons) > 0

    def test_case_insensitive_genre_match(self):
        user = {"genre": "POP", "mood": "", "energy": 0.5}
        song = {"genre": "pop", "mood": "", "energy": 0.5}

        score_with,    _ = score_song(user, song)
        user_no = {"genre": "POP", "mood": "", "energy": 0.5}
        song_no = {"genre": "jazz", "mood": "", "energy": 0.5}
        score_without, _ = score_song(user_no, song_no)

        assert score_with > score_without

    def test_empty_user_prefs_still_returns_score(self):
        user = {"genre": "", "mood": "", "energy": 0.5}
        song = {"genre": "pop", "mood": "happy", "energy": 0.5}

        score, reasons = score_song(user, song)

        assert isinstance(score, float)
        assert score >= 0.0


# ── recommend_songs (functional API) ─────────────────────────────────────────

class TestRecommendSongs:
    def test_returns_top_k_results(self):
        user = {"genre": "pop", "mood": "happy", "energy": 0.9}
        songs = [s.__dict__ for s in ALL_SONGS]

        results = recommend_songs(user, songs, k=2)

        assert len(results) == 2

    def test_results_sorted_by_score_descending(self):
        user = {"genre": "pop", "mood": "happy", "energy": 0.9}
        songs = [s.__dict__ for s in ALL_SONGS]

        results = recommend_songs(user, songs, k=3)
        scores = [r[1] for r in results]

        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_catalog_returns_all(self):
        user = {"genre": "pop", "mood": "happy", "energy": 0.9}
        songs = [s.__dict__ for s in ALL_SONGS]

        results = recommend_songs(user, songs, k=100)

        assert len(results) == len(ALL_SONGS)

    def test_each_result_is_tuple_of_song_score_reasons(self):
        user = {"genre": "pop", "mood": "happy", "energy": 0.9}
        songs = [POP_HAPPY_HIGH.__dict__]

        results = recommend_songs(user, songs, k=1)
        song, score, reasons = results[0]

        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(reasons, list)


# ── Recommender class ─────────────────────────────────────────────────────────

class TestRecommender:
    def test_recommend_returns_correct_count(self):
        user = UserProfile("pop", "happy", 0.9, False)
        rec = make_recommender()

        results = rec.recommend(user, k=2)

        assert len(results) == 2

    def test_recommend_top_result_matches_user_profile(self):
        user = UserProfile("pop", "happy", 0.9, False)
        rec = make_recommender()

        results = rec.recommend(user, k=3)

        assert results[0].genre == "pop"
        assert results[0].mood == "happy"

    def test_recommend_sorted_by_score(self):
        user = UserProfile("lofi", "chill", 0.3, True)
        rec = make_recommender()

        results = rec.recommend(user, k=3)

        # Verify ranking matches expectations based on lofi/chill/low-energy profile
        assert results[0].genre == "lofi"

    def test_recommend_k_of_one_returns_best_match(self):
        user = UserProfile("rock", "intense", 0.7, False)
        rec = make_recommender()

        results = rec.recommend(user, k=1)

        assert len(results) == 1
        assert results[0].genre == "rock"

    def test_explain_recommendation_returns_string(self):
        user = UserProfile("pop", "happy", 0.9, False)
        rec = make_recommender()

        explanation = rec.explain_recommendation(user, POP_HAPPY_HIGH)

        assert isinstance(explanation, str)
        assert explanation.strip() != ""

    def test_explain_recommendation_mentions_genre_match(self):
        user = UserProfile("pop", "happy", 0.9, False)
        rec = make_recommender()

        explanation = rec.explain_recommendation(user, POP_HAPPY_HIGH)

        assert "genre" in explanation.lower()

    def test_explain_recommendation_mentions_mood_match(self):
        user = UserProfile("pop", "happy", 0.9, False)
        rec = make_recommender()

        explanation = rec.explain_recommendation(user, POP_HAPPY_HIGH)

        assert "mood" in explanation.lower()


# ── load_songs ────────────────────────────────────────────────────────────────

class TestLoadSongs:
    def _write_temp_csv(self, rows):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "title", "artist", "genre", "mood",
                "energy", "tempo_bpm", "valence", "danceability", "acousticness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        return f.name

    def test_loads_correct_number_of_songs(self):
        path = self._write_temp_csv([
            dict(id=1, title="A", artist="X", genre="pop", mood="happy",
                 energy=0.8, tempo_bpm=120, valence=0.8, danceability=0.7, acousticness=0.1),
            dict(id=2, title="B", artist="Y", genre="lofi", mood="chill",
                 energy=0.3, tempo_bpm=75, valence=0.5, danceability=0.4, acousticness=0.9),
        ])
        try:
            songs = load_songs(path)
            assert len(songs) == 2
        finally:
            os.unlink(path)

    def test_energy_is_parsed_as_float(self):
        path = self._write_temp_csv([
            dict(id=1, title="A", artist="X", genre="pop", mood="happy",
                 energy=0.75, tempo_bpm=120, valence=0.8, danceability=0.7, acousticness=0.1),
        ])
        try:
            songs = load_songs(path)
            assert isinstance(songs[0]["energy"], float)
        finally:
            os.unlink(path)

    def test_loads_real_catalog(self):
        songs = load_songs("data/songs.csv")
        assert len(songs) > 0
        assert "title" in songs[0]
        assert "genre" in songs[0]
        assert isinstance(songs[0]["energy"], float)
