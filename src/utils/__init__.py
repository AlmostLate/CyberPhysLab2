"""
Utils module for Lab 2 NLP.
"""

from src.utils.data_loader import (
    load_adult_dataset,
    preprocess_dataset,
    prepare_train_test_split
)

__all__ = [
    "load_adult_dataset",
    "preprocess_dataset",
    "prepare_train_test_split"
]
