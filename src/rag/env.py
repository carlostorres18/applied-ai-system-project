import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    explicit_path = os.getenv("OMNI_DOTENV_PATH")
    if explicit_path:
        load_dotenv(dotenv_path=explicit_path, override=False)

    project_root = Path(__file__).resolve().parents[2]

    candidates = [
        project_root / ".env",
        Path.cwd() / ".env",
        Path.home() / ".config" / "omni_code" / ".env",
        Path.home() / ".config" / "omni" / ".env",
    ]

    loaded_any = False
    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
            loaded_any = True

    if not loaded_any:
        load_dotenv(override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
