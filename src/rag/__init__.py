"""
RAG (Retrieval-Augmented Generation) module for Lab 2 NLP.

This module provides:
- Sentence transformer embeddings
- FAISS vector indexing
- RAG retrieval
"""

from src.rag.embedder import SentenceEmbedder
from src.rag.indexer import FAISSIndexer
from src.rag.retriever import RAGRetriever

__all__ = ["SentenceEmbedder", "FAISSIndexer", "RAGRetriever"]
