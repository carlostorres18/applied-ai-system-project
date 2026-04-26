from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initializes the recommender with a list of songs."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top-k songs ranked for the given user profile."""
        ranked_songs = sorted(
            self.songs,
            key=lambda song: score_song(
                {
                    "genre": user.favorite_genre,
                    "mood": user.favorite_mood,
                    "energy": user.target_energy,
                    "likes_acoustic": user.likes_acoustic,
                },
                {
                    "id": song.id,
                    "title": song.title,
                    "artist": song.artist,
                    "genre": song.genre,
                    "mood": song.mood,
                    "energy": song.energy,
                    "tempo_bpm": song.tempo_bpm,
                    "valence": song.valence,
                    "danceability": song.danceability,
                    "acousticness": song.acousticness,
                },
            )[0],
            reverse=True,
        )
        return ranked_songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Builds a short explanation for why a song was recommended."""
        _, reasons = score_song(
            {
                "genre": user.favorite_genre,
                "mood": user.favorite_mood,
                "energy": user.target_energy,
                "likes_acoustic": user.likes_acoustic,
            },
            {
                "id": song.id,
                "title": song.title,
                "artist": song.artist,
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "tempo_bpm": song.tempo_bpm,
                "valence": song.valence,
                "danceability": song.danceability,
                "acousticness": song.acousticness,
            },
        )
        return "; ".join(reasons)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song and returns an explainable reason list.

    Scoring recipe:
    - +2.0 for genre match
    - +1.5 for mood match
    - Up to +2.0 for energy proximity (closer is better)
    """
    score = 0.0
    reasons: List[str] = []

    target_genre = str(user_prefs.get("genre", "")).strip().lower()
    target_mood = str(user_prefs.get("mood", "")).strip().lower()
    target_energy = float(user_prefs.get("energy", 0.5))

    song_genre = str(song.get("genre", "")).strip().lower()
    song_mood = str(song.get("mood", "")).strip().lower()
    song_energy = float(song.get("energy", 0.0))

    if target_genre and song_genre == target_genre:
        genre_points = 1.0
        score += genre_points
        reasons.append(f"genre match (+{genre_points:.1f})")

    if target_mood and song_mood == target_mood:
        mood_points = 1.5
        score += mood_points
        reasons.append(f"mood match (+{mood_points:.1f})")

    energy_delta = abs(song_energy - target_energy)
    energy_points = max(0.0, 4.0 - (energy_delta * 4.0))
    score += energy_points
    reasons.append(
        f"energy closeness (+{energy_points:.2f}, target={target_energy:.2f}, song={song_energy:.2f})"
    )

    if not reasons:
        reasons.append("baseline match (+0.0)")

    return score, reasons

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            song: Dict = {
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            }
            songs.append(song)

    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, List[str]]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored: List[Tuple[Dict, float, List[str]]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, reasons))

    scored.sort(key=lambda rec: rec[1], reverse=True)
    return scored[:k]
