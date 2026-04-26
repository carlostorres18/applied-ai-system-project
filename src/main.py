import argparse
import json
from typing import Any, Dict, Optional

from src.recommender import load_songs, recommend_songs
from src.system import get_playlist_agent, get_search_engine, load_user_profile_stub


def main() -> None:
    parser = argparse.ArgumentParser(prog="music-recommender")
    sub = parser.add_subparsers(dest="cmd")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=5)

    playlist = sub.add_parser("playlist")
    playlist.add_argument("goal")
    playlist.add_argument("--minutes", type=int, default=45)
    playlist.add_argument("--unique-artist", action="store_true", default=False)

    legacy = sub.add_parser("legacy")
    legacy.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "search":
        _run_search(query=args.query, k=args.k)
        return

    if args.cmd == "playlist":
        _run_playlist(goal=args.goal, minutes=args.minutes, unique_artist=args.unique_artist)
        return

    _run_legacy(k=getattr(args, "k", 5))


def _run_search(query: str, k: int) -> None:
    engine = get_search_engine()
    user_profile = load_user_profile_stub()
    results = engine.search(query, user_profile=user_profile, k=k)
    payload = [r.__dict__ for r in results]
    print(json.dumps(payload, indent=2))


def _run_playlist(goal: str, minutes: int, unique_artist: bool) -> None:
    agent = get_playlist_agent()
    user_profile = load_user_profile_stub()
    constraints: Dict[str, Any] = {"unique_artist": unique_artist}
    plan = agent.build_playlist(goal=goal, user_profile=user_profile, minutes=minutes, constraints=constraints)
    payload = {
        "title": plan.title,
        "goal": plan.goal,
        "minutes": plan.minutes,
        "trace": plan.trace,
        "items": [{"song": i.song, "explanation": i.explanation} for i in plan.items],
    }
    print(json.dumps(payload, indent=2))


def _run_legacy(k: int) -> None:
    songs = load_songs("data/songs.csv")
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=k)

    print("\n" + "=" * 72)
    print("TOP RECOMMENDATIONS (LEGACY SCORING)")
    print("=" * 72)

    for index, rec in enumerate(recommendations, start=1):
        song, score, reasons = rec
        print(f"\n[{index}] {song['title']}")
        print(f"    Final Score : {score:.2f}")
        print("    Reasons     :")

        if isinstance(reasons, list) and reasons:
            for reason in reasons:
                print(f"      - {reason}")
        elif isinstance(reasons, str) and reasons.strip():
            print(f"      - {reasons}")
        else:
            print("      - No specific reasons provided")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
