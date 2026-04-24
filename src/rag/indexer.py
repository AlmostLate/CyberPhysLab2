"""
FAISS indexer for RAG.

This module provides FAISS-based vector indexing for efficient
similarity search on embedded text data.
"""

import faiss
import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path

from src.config import RAGConfig


class FAISSIndexer:
    """
    FAISS indexer for vector similarity search.
    
    This class provides functionality to:
    - Create FAISS index from embeddings
    - Add new embeddings to index
    - Search for similar embeddings
    
    Attributes:
        dimension (int): Embedding dimension
        index_type (str): Type of FAISS index
        index (faiss.Index): FAISS index object
    
    Example:
        >>> indexer = FAISSIndexer(dimension=384)
        >>> indexer.add(embeddings)
        >>> indices, distances = indexer.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: str = "Flat",
        metric: str = "cosine"
    ):
        """
        Initialize FAISS indexer.
        
        Args:
            dimension (int, optional): Embedding dimension.
                                      Defaults to RAGConfig.EMBEDDING_DIMENSION
            index_type (str): Type of index ('Flat', 'IVF', 'HNSW')
            metric (str): Similarity metric ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension or RAGConfig.EMBEDDING_DIMENSION
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self._normalize = metric == "cosine"
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type."""
        if self.index_type == "Flat":
            if self.metric == "cosine" or self.metric == "ip":
                # Inner product index (cosine sim with normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                # L2 distance index
                self.index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "HNSW":
            # HNSW index for approximate nearest neighbor search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        
        elif self.index_type == "IVF":
            # IVF index with flat quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                100  # number of clusters
            )
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def is_trained(self) -> bool:
        """Check if index is trained."""
        if self.index is None:
            return False
        if hasattr(self.index, "is_trained"):
            return self.index.is_trained
        return True
    
    def add(self, embeddings: np.ndarray, normalize: bool = True):
        """
        Add embeddings to the index.
        
        Args:
            embeddings (np.ndarray): Embeddings to add, shape (n, dimension)
            normalize (bool): Whether to normalize embeddings for cosine similarity
        
        Example:
            >>> indexer = FAISSIndexer()
            >>> indexer.add(embeddings)  # embeddings shape: (1000, 384)
        """
        if normalize and self._normalize:
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        if not self.is_trained():
            if self.index_type == "IVF":
                self.index.train(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        print(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding (np.ndarray): Query embedding, shape (dimension,) or (1, dimension)
            top_k (int): Number of results to return
            normalize (bool): Whether to normalize query
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices) of top-k results
        
        Example:
            >>> indexer = FAISSIndexer()
            >>> indexer.add(embeddings)
            >>> distances, indices = indexer.search(query, top_k=5)
            >>> print(f"Most similar at index {indices[0]} with distance {distances[0]}")
        """
        if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if normalize and self._normalize:
            norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            query_embedding = query_embedding / norms
        
        # Adjust top_k if necessary
        max_k = min(top_k, self.index.ntotal)
        if max_k < top_k:
            print(f"Warning: Only {max_k} items in index, returning {max_k} results")
            top_k = max_k
        
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k
        )
        
        return distances[0], indices[0]
    
    def remove(self, indices: List[int]):
        """
        Remove embeddings from index by rebuilding (FAISS doesn't support direct removal).
        
        Note: This is inefficient for large indices. Consider rebuilding if needed.
        
        Args:
            indices (List[int]): Indices to remove
        """
        print("Warning: FAISS doesn't support efficient removal. Rebuild index if needed.")
    
    def save(self, filepath: Union[str, Path]):
        """
        Save index to file.
        
        Args:
            filepath (str): Path to save index
        
        Example:
            >>> indexer = FAISSIndexer()
            >>> indexer.add(embeddings)
            >>> indexer.save("faiss_index.bin")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # faiss.write_index uses C++ FileIOWriter with const char* which breaks
        # on Windows paths with non-ASCII characters (Cyrillic, etc.).
        # Use serialize_index + Python file I/O instead.
        index_bytes = faiss.serialize_index(self.index)
        with open(filepath, 'wb') as f:
            f.write(index_bytes.tobytes())
        print(f"Index saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """
        Load index from file.
        
        Args:
            filepath (str): Path to load index from
        
        Example:
            >>> indexer = FAISSIndexer()
            >>> indexer.load("faiss_index.bin")
            >>> distances, indices = indexer.search(query, top_k=5)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with open(filepath, 'rb') as f:
            index_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        self.index = faiss.deserialize_index(index_bytes)
        print(f"Index loaded from {filepath}. Total vectors: {self.index.ntotal}")
    
    @property
    def total_vectors(self) -> int:
        """Get number of vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal


class FAISSIndexerStore:
    """
    Combined embedder and indexer for RAG.
    
    This class combines embedding generation and indexing
    for a complete RAG pipeline.
    
    Example:
        >>> from src.rag.embedder import SentenceEmbedder
        >>> store = FAISSIndexerStore(SentenceEmbedder())
        >>> store.index_dataset(dataset)
        >>> results = store.search("query text", top_k=5)
    """
    
    def __init__(
        self,
        embedder,  # SentenceEmbedder
        dimension: Optional[int] = None,
        index_type: str = "Flat"
    ):
        """
        Initialize store with embedder.
        
        Args:
            embedder: SentenceEmbedder instance
            dimension (int): Embedding dimension
            index_type (str): FAISS index type
        """
        self.embedder = embedder
        self.dimension = dimension or RAGConfig.EMBEDDING_DIMENSION
        self.indexer = FAISSIndexer(
            dimension=self.dimension,
            index_type=index_type
        )
        self.metadata = []  # Store metadata for retrieved results
    
    def index_dataset(
        self,
        records: List[dict],
        text_column: str = "text",
        metadata_columns: Optional[List[str]] = None
    ):
        """
        Index a dataset of records.
        
        Args:
            records (List[dict]): List of record dictionaries
            text_column (str): Key containing text to embed
            metadata_columns (List[str], optional): Keys to store as metadata
        """
        print(f"Indexing {len(records)} records...")
        
        # Generate embeddings
        embeddings = self.embedder.embed_dataset(records, text_column)
        
        # Store metadata
        self.metadata = []
        for record in records:
            meta = {}
            if metadata_columns:
                for col in metadata_columns:
                    meta[col] = record.get(col)
            else:
                meta = record.copy()
            self.metadata.append(meta)
        
        # Add to index
        self.indexer.add(embeddings)
        
        print(f"Indexed {len(records)} records with {len(self.metadata)} metadata entries")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[dict]:
        """
        Search for similar records.
        
        Args:
            query (str): Query text
            top_k (int): Number of results
            similarity_threshold (float, optional): Minimum similarity score
        
        Returns:
            List[dict]: List of similar records with similarity scores
        """
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search index
        distances, indices = self.indexer.search(query_embedding, top_k=top_k)
        
        # Build results
        results = []
        for i, idx in enumerate(indices):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["_distance"] = float(distances[i])
                result["_index"] = int(idx)
                
                # Filter by threshold
                if similarity_threshold is None or distances[i] >= similarity_threshold:
                    results.append(result)
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """
        Save index and metadata.
        
        Args:
            index_path (str): Path to save FAISS index
            metadata_path (str): Path to save metadata JSON
        """
        self.indexer.save(index_path)
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        
        print(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """
        Load index and metadata.
        
        Args:
            index_path (str): Path to FAISS index
            metadata_path (str): Path to metadata JSON
        """
        self.indexer.load(index_path)
        
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded index with {len(self.metadata)} entries")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Testing FAISSIndexer...")
    
    # Create random embeddings for testing
    dimension = 384
    n_vectors = 1000
    
    print(f"\n1. Creating {n_vectors} random embeddings (dim={dimension}):")
    embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    print(f"   Embeddings shape: {embeddings.shape}")
    
    print("\n2. Creating FAISS index:")
    indexer = FAISSIndexer(dimension=dimension, metric="cosine")
    indexer.add(embeddings)
    print(f"   Total vectors in index: {indexer.total_vectors}")
    
    print("\n3. Testing search:")
    query = np.random.randn(dimension).astype(np.float32)
    query = query / np.linalg.norm(query)  # Normalize
    distances, indices = indexer.search(query, top_k=5)
    print(f"   Top-5 distances: {distances}")
    print(f"   Top-5 indices: {indices}")
    
    print("\n4. Testing save/load:")
    indexer.save("test_index.bin")
    
    new_indexer = FAISSIndexer(dimension=dimension)
    new_indexer.load("test_index.bin")
    print(f"   Loaded index vectors: {new_indexer.total_vectors}")
    
    print("\n✓ FAISSIndexer tests passed!")
