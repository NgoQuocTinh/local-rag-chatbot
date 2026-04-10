"""LLM provider abstraction - Ollama, Gemini, etc."""

from src.llm.llm_factory import get_llm

__all__ = ["get_llm"]
