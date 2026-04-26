from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SongDoc:
    song_id: int
    text: str
    metadata: Dict[str, Any]


def build_song_doc(song: Dict[str, Any], extra_context: Optional[str] = None) -> SongDoc:
    parts = [
        f"Title: {song.get('title', '')}",
        f"Artist: {song.get('artist', '')}",
        f"Genre: {song.get('genre', '')}",
        f"Mood: {song.get('mood', '')}",
        f"Energy: {song.get('energy', '')}",
        f"Tempo BPM: {song.get('tempo_bpm', '')}",
        f"Valence: {song.get('valence', '')}",
        f"Danceability: {song.get('danceability', '')}",
        f"Acousticness: {song.get('acousticness', '')}",
    ]

    if extra_context:
        parts.append(f"Context: {extra_context}")

    text = "\n".join(parts)
    return SongDoc(song_id=int(song["id"]), text=text, metadata=song)
