"""
RAG retriever for credit scoring.

This module provides RAG (Retrieval-Augmented Generation) retrieval
functionality for the Adult Income dataset.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.rag.embedder import SentenceEmbedder
from src.rag.indexer import FAISSIndexer, FAISSIndexerStore
from src.config import RAGConfig, DatasetConfig, DATA_DIR


class RAGRetriever:
    """
    RAG retriever for credit scoring cases.
    
    This class provides retrieval functionality to find similar
    historical credit cases based on client descriptions.
    
    Attributes:
        embedder (SentenceEmbedder): Text embedder
        indexer_store (FAISSIndexerStore): Combined indexer and metadata store
    
    Example:
        >>> retriever = RAGRetriever()
        >>> retriever.index_dataset(df)
        >>> results = retriever.retrieve("High income married client", top_k=5)
    """
    
    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        index_type: str = "Flat"
    ):
        """
        Initialize RAG retriever.
        
        Args:
            embedder (SentenceEmbedder, optional): Custom embedder
            index_type (str): FAISS index type
        """
        self.embedder = embedder or SentenceEmbedder()
        self.indexer_store = FAISSIndexerStore(
            embedder=self.embedder,
            dimension=RAGConfig.EMBEDDING_DIMENSION,
            index_type=index_type
        )
        self.is_indexed = False
    
    def create_text_representation(self, row: pd.Series) -> str:
        """
        Create text representation of a data row.
        
        Args:
            row (pd.Series): Data row
        
        Returns:
            str: Text representation
        """
        parts = []
        
        # Demographics
        if "age" in row:
            parts.append(f"Age {int(row['age'])}")
        
        if "workclass" in row:
            parts.append(f"Workclass: {row['workclass'].strip()}")
        
        if "education" in row:
            parts.append(f"Education: {row['education'].strip()}")
        
        # Financial
        if "income" in row:
            parts.append(f"Income: {row['income'].strip()}")
        
        if "capital_gain" in row and row.get("capital_gain", 0) > 0:
            parts.append(f"Capital gain: ${row['capital_gain']}")
        
        if "capital_loss" in row and row.get("capital_loss", 0) > 0:
            parts.append(f"Capital loss: ${row['capital_loss']}")
        
        # Employment
        if "hours_per_week" in row:
            parts.append(f"Works {int(row['hours_per_week'])} hours/week")
        
        if "occupation" in row:
            parts.append(f"Occupation: {row['occupation'].strip()}")
        
        # Family
        if "marital_status" in row:
            parts.append(f"Marital status: {row['marital_status'].strip()}")
        
        if "relationship" in row:
            parts.append(f"Relationship: {row['relationship'].strip()}")
        
        return ", ".join(parts)
    
    def index_dataset(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None,
        save_index: bool = True
    ):
        """
        Index a dataset for retrieval.
        
        Args:
            df (pd.DataFrame): Dataset to index
            text_column (str, optional): Column containing text.
                                        If None, creates from other columns.
            save_index (bool): Whether to save index to disk
        """
        print(f"Indexing dataset with {len(df)} records...")
        
        # Create records
        records = []
        for idx, row in df.iterrows():
            if text_column and text_column in row:
                text = row[text_column]
            else:
                text = self.create_text_representation(row)
            
            record = {
                "text": text,
                "index": idx
            }
            
            # Add all columns as metadata
            for col in df.columns:
                record[col] = row[col]
            
            records.append(record)
        
        # Index dataset
        self.indexer_store.index_dataset(
            records,
            text_column="text"
        )
        
        self.is_indexed = True
        
        if save_index:
            self.save_index()
        
        print(f"Indexed {len(records)} records")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar cases for a query.
        
        Args:
            query (str): Query describing a client
            top_k (int): Number of results to retrieve
            similarity_threshold (float, optional): Minimum similarity score
            include_metadata (bool): Whether to include full metadata
        
        Returns:
            List[Dict[str, Any]]: List of similar cases with scores
        
        Example:
            >>> retriever = RAGRetriever()
            >>> retriever.index_dataset(df)
            >>> results = retriever.retrieve("35 year old married professional", top_k=5)
            >>> for r in results:
            ...     print(f"Income: {r.get('income')}, Similarity: {r.get('_distance'):.3f}")
        """
        if not self.is_indexed:
            print("Warning: Index not created. Loading from disk...")
            try:
                self.load_index()
            except:
                print("Error: No index found. Please index a dataset first.")
                return []
        
        results = self.indexer_store.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Filter and format results
        formatted_results = []
        for r in results:
            result = {
                "_distance": r.get("_distance", 0),
                "_index": r.get("_index", -1),
                "text": r.get("text", "")
            }
            
            if include_metadata:
                # Add original columns
                for key, value in r.items():
                    if not key.startswith("_") and key != "text":
                        result[key] = value
            
            formatted_results.append(result)
        
        return formatted_results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve similar cases and format as context string.
        
        Args:
            query (str): Query describing a client
            top_k (int): Number of results
        
        Returns:
            str: Formatted context string for LLM
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return "No similar historical cases found."
        
        context_parts = ["Similar historical cases:\n"]
        
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"Case {i} (similarity: {r['_distance']:.3f}):\n"
                f"  {r['text']}\n"
                f"  Outcome: {r.get('income', 'Unknown')}"
            )
        
        return "\n".join(context_parts)
    
    def save_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Save index and metadata to disk.
        
        Args:
            index_path (str, optional): Path for FAISS index
            metadata_path (str, optional): Path for metadata JSON
        """
        index_path = index_path or str(Path(DATA_DIR) / "faiss_index" / "index.faiss")
        metadata_path = metadata_path or str(Path(DATA_DIR) / "faiss_index" / "metadata.json")
        
        self.indexer_store.save(index_path, metadata_path)
    
    def load_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Load index and metadata from disk.
        
        Args:
            index_path (str, optional): Path for FAISS index
            metadata_path (str, optional): Path for metadata JSON
        """
        index_path = index_path or str(Path(DATA_DIR) / "faiss_index" / "index.faiss")
        metadata_path = metadata_path or str(Path(DATA_DIR) / "faiss_index" / "metadata.json")
        
        self.indexer_store.load(index_path, metadata_path)
        self.is_indexed = True


def get_default_retriever() -> RAGRetriever:
    """
    Get a default RAG retriever.
    
    Returns:
        RAGRetriever: Default retriever instance
    """
    return RAGRetriever()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("RAG Retriever module loaded successfully")
    print("Usage:")
    print("  retriever = RAGRetriever()")
    print("  retriever.index_dataset(df)")
    print("  results = retriever.retrieve('query text', top_k=5)")
