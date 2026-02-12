import os
from typing import Optional

# New imports
from core.llm_client import BaseLLMClient, GeminiClient, OllamaClient


def get_api_key(explicit: Optional[str] = None, allow_missing: bool = False) -> str:
    key = explicit or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    # Only enforce key presence if explicitly asking for Gemini or default behavior
    if not key and not allow_missing:
        # We might not need to raise here if switching to Ollama, but keeping for backward compat
        pass 
    return key or ""


def get_llm_client(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434/v1",
    ollama_model: str = "qwen3:8b"
) -> BaseLLMClient:
    """
    Factory function to get the appropriate LLM client.
    
    Args:
        provider: "gemini" or "ollama"
        api_key: Google API Key (required for Gemini)
        ollama_base_url: URL for Ollama API
        ollama_model: Model name for Ollama
    """
    if provider.lower() == "ollama":
        return OllamaClient(base_url=ollama_base_url, model_name=ollama_model)
    
    # Default to Gemini
    key = get_api_key(api_key, allow_missing=(provider.lower() == "ollama"))
    if not key and provider.lower() == "gemini":
         raise ValueError("Google API Key is required for Gemini provider.")
         
    return GeminiClient(api_key=key)

# Deprecated but kept for compatibility during migration if needed
def get_client(api_key: Optional[str] = None) -> BaseLLMClient:
    return get_llm_client(provider="gemini", api_key=api_key)

