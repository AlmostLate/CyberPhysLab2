"""
Sentence transformer embedder for RAG.

This module provides text embedding functionality using sentence transformers
for the Adult Income dataset (credit scoring).
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import numpy as np
from pathlib import Path

from src.config import RAGConfig


class SentenceEmbedder:
    """
    Sentence transformer embedder for text embeddings.
    
    This class provides functionality to generate embeddings for text
    using pre-trained sentence transformer models.
    
    Attributes:
        model_name (str): Name of the sentence transformer model
        model (SentenceTransformer): The underlying sentence transformer model
    
    Example:
        >>> embedder = SentenceEmbedder()
        >>> embedding = embedder.embed("High income, married, bachelor's degree")
        >>> embeddings = embedder.embed_batch(["Text 1", "Text 2"])
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize sentence embedder.
        
        Args:
            model_name (str, optional): Name of sentence transformer model.
                                        Defaults to RAGConfig.EMBEDDING_MODEL
            device (str, optional): Device to run model on ('cpu', 'cuda').
                                   Defaults to auto-detect.
        """
        self.model_name = model_name or RAGConfig.EMBEDDING_MODEL
        self.device = device
        
        print(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Model loaded successfully. Embedding dimension: {self.dimension}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return RAGConfig.EMBEDDING_DIMENSION
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to embed
        
        Returns:
            np.ndarray: Embedding vector of shape (embedding_dimension,)
        
        Example:
            >>> embedder = SentenceEmbedder()
            >>> vec = embedder.embed("A client with high income")
            >>> print(vec.shape)  # (384,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            show_progress (bool): Whether to show progress bar
        
        Returns:
            np.ndarray: Embeddings of shape (len(texts), embedding_dimension)
        
        Example:
            >>> embedder = SentenceEmbedder()
            >>> vecs = embedder.embed_batch(["Text 1", "Text 2", "Text 3"])
            >>> print(vecs.shape)  # (3, 384)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def embed_dataset(
        self,
        records: List[dict],
        text_column: str = "text",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a dataset.
        
        Args:
            records (List[dict]): List of record dictionaries
            text_column (str): Key in records containing text to embed
            show_progress (bool): Whether to show progress bar
        
        Returns:
            np.ndarray: Embeddings of shape (len(records), embedding_dimension)
        """
        texts = [record.get(text_column, "") for record in records]
        return self.embed_batch(texts, show_progress=show_progress)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: Union[str, Path]) -> None:
        """
        Save embeddings to file.
        
        Args:
            embeddings (np.ndarray): Embeddings to save
            filepath (str): Path to save embeddings
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, embeddings)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: Union[str, Path]) -> np.ndarray:
        """
        Load embeddings from file.
        
        Args:
            filepath (str): Path to load embeddings from
        
        Returns:
            np.ndarray: Loaded embeddings
        """
        return np.load(filepath)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
        
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> tuple:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            candidate_embeddings (np.ndarray): Candidate embeddings to search
            top_k (int): Number of results to return
        
        Returns:
            tuple: (indices, scores) - indices of top-k similar items and their scores
        """
        # Compute similarities
        similarities = np.dot(candidate_embeddings, query_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores


def get_default_embedder() -> SentenceEmbedder:
    """
    Get a default sentence embedder with configured settings.
    
    Returns:
        SentenceEmbedder: Embedder with default configuration
    """
    return SentenceEmbedder()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Testing SentenceEmbedder...")
    
    embedder = SentenceEmbedder()
    
    # Test single embedding
    print("\n1. Testing single text embedding:")
    text = "High income, married, with bachelor's degree"
    embedding = embedder.embed(text)
    print(f"   Text: {text}")
    print(f"   Embedding shape: {embedding.shape}")
    
    # Test batch embedding
    print("\n2. Testing batch embedding:")
    texts = [
        "High income, married, bachelor's degree",
        "Low income, single, high school",
        "Medium income, divorced, master's degree"
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"   Number of texts: {len(texts)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    print("\n3. Testing similarity computation:")
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"   Similarity between text 0 and 1: {sim:.4f}")
    
    print("\n✓ SentenceEmbedder tests passed!")
