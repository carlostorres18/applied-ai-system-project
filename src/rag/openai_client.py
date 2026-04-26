from openai import OpenAI

from src.rag.env import load_env, require_env


def get_openai_client() -> OpenAI:
    load_env()
    api_key = require_env("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)
