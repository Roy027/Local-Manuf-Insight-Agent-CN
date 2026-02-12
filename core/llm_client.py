import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

# Try importing google.genai, but don't fail if strictly using Ollama (though dependencies might require it)
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from openai import OpenAI


class LLMResponse:
    """Standardized response object for LLM calls."""
    def __init__(self, text: str):
        self.text = text


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Any] = None
    ) -> LLMResponse:
        """
        Generate content from the LLM.
        
        Args:
            prompt: The text prompt.
            response_mime_type: Expected MIME type (e.g., "application/json").
            response_schema: Optional schema for structured output (e.g., Pydantic model or schema dict).
            
        Returns:
            LLMResponse object containing the text.
        """
        pass


class GeminiClient(BaseLLMClient):
    """Wrapper for Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        if not genai:
            raise ImportError("google-genai package is not installed.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Any] = None
    ) -> LLMResponse:
        config = None
        if response_mime_type or response_schema:
            config = types.GenerateContentConfig(
                response_mime_type=response_mime_type,
                response_schema=response_schema
            )
            config.response_schema = response_schema
        
        # Note: google-genai client handles system_instruction via model config or separate argument depending on version.
        # For simplicity in this wrapper, we prepend validation or context if strictly needed, 
        # but the modern client usually supports it in GenerateContentConfig or method arg.
        # Let's check if 'system_instruction' is valid for this client version, otherwise prepend.
        if system_instruction:
             config = config or types.GenerateContentConfig()
             config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return LLMResponse(text=response.text or "")


class OllamaClient(BaseLLMClient):
    """Wrapper for Ollama (via OpenAI-compatible API)."""
    
    def __init__(self, base_url: str, model_name: str, api_key: str = "ollama"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name

    def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Any] = None
    ) -> LLMResponse:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        # Handle JSON mode for Ollama/Qwen
        response_format = None
        if response_mime_type == "application/json":
            # Append instruction to prompt if not already there (simple heursitic)
            if "json" not in prompt.lower():
                messages[0]["content"] += "\n\nPlease respond in valid JSON format."
            
            # OpenAI compatible way to request JSON
            response_format = {"type": "json_object"}

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
            )
            return LLMResponse(text=chat_completion.choices[0].message.content or "")
        except Exception as e:
            # Fallback or re-raise
            return LLMResponse(text=f"Error generating content: {str(e)}")
