"""
LLM module for Lab 2 NLP - Ollama integration.
"""

from src.llm.ollama_client import OllamaClient
from src.llm.service import create_app, run_service

__all__ = ["OllamaClient", "create_app", "run_service"]
