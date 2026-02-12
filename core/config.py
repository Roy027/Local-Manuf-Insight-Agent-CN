import os
from typing import Optional

from google import genai


def get_api_key(explicit: Optional[str] = None, allow_missing: bool = False) -> str:
    key = explicit or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key and not allow_missing:
        raise RuntimeError("Google API Key not provided. Set GOOGLE_API_KEY or pass it in.")
    return key or ""


def get_client(api_key: Optional[str] = None) -> genai.Client:
    """Create a Google GenAI client using provided or env key."""
    key = get_api_key(api_key)
    return genai.Client(api_key=key)
