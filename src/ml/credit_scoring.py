"""
Credit scoring ML models.

This module provides traditional machine learning models for credit scoring
including Logistic Regression, Random Forest, and Gradient Boosting.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pickle
from pathlib import Path

from src.config import MLConfig, MODELS_DIR


class CreditScoringModel:
    """
    Base class for credit scoring models.
    
    This class provides common functionality for credit scoring ML models.
    
    Attributes:
        model: The underlying sklearn model
        model_name (str): Name of the model
        is_trained (bool): Whether the model has been trained
    """
    
    def __init__(self, model_name: str):
        """
        Initialize credit scoring model.
        
        Args:
            model_name (str): Name of the model
        """
        self.model = None
        self.model_name = model_name
        self.is_trained = False
        self.feature_names = None
        self.classes_ = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            **kwargs: Additional model-specific arguments
        
        Returns:
            Dict[str, float]: Training metrics
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features to predict
        
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0)
        }
        
        return metrics
    
    def get_confusion_matrix(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> np.ndarray:
        """Get confusion matrix."""
        y_pred = self.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    
    def get_classification_report(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> str:
        """Get classification report."""
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def save(self, filepath: str):
        """Save model to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "model_name": self.model_name,
                "feature_names": self.feature_names,
                "classes_": self.classes_
            }, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.model_name = data["model_name"]
        self.feature_names = data.get("feature_names")
        self.classes_ = data.get("classes_")
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


class LogisticRegressionModel(CreditScoringModel):
    """
    Logistic Regression for credit scoring.
    
    A simple, interpretable model that outputs probabilities.
    
    Example:
        >>> model = LogisticRegressionModel()
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(**MLConfig.LOGISTIC_REGRESSION_PARAMS)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the logistic regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.feature_names = X_train.columns.tolist()
        self.classes_ = y_train.unique()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        y_train_pred = self.predict(X_train)
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred)
        }
        
        return train_metrics
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get feature coefficients.
        
        Returns:
            pd.DataFrame: Feature names and their coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        coef_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_[0]
        })
        return coef_df.sort_values("coefficient", ascending=False)


class RandomForestModel(CreditScoringModel):
    """
    Random Forest for credit scoring.
    
    An ensemble model that handles non-linear relationships well.
    
    Example:
        >>> model = RandomForestModel()
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(**MLConfig.RANDOM_FOREST_PARAMS)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the random forest model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.feature_names = X_train.columns.tolist()
        self.classes_ = y_train.unique()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_train_pred = self.predict(X_train)
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred)
        }
        
        return train_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: Feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        })
        return importance_df.sort_values("importance", ascending=False)


class GradientBoostingModel(CreditScoringModel):
    """
    Gradient Boosting for credit scoring.
    
    A powerful ensemble method that builds trees sequentially.
    
    Example:
        >>> model = GradientBoostingModel()
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(**MLConfig.GRADIENT_BOOSTING_PARAMS)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the gradient boosting model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.feature_names = X_train.columns.tolist()
        self.classes_ = y_train.unique()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_train_pred = self.predict(X_train)
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred)
        }
        
        return train_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: Feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        })
        return importance_df.sort_values("importance", ascending=False)


# =============================================================================
# MODEL FACTORY
# =============================================================================
def get_model(model_type: str) -> CreditScoringModel:
    """
    Get a credit scoring model by type.
    
    Args:
        model_type (str): Type of model ('logistic_regression', 'random_forest', 'gradient_boosting')
    
    Returns:
        CreditScoringModel: Initialized model
    
    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower()
    
    if model_type == "logistic_regression":
        return LogisticRegressionModel()
    elif model_type == "random_forest":
        return RandomForestModel()
    elif model_type == "gradient_boosting":
        return GradientBoostingModel()
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: 'logistic_regression', 'random_forest', 'gradient_boosting'"
        )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_types: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple models.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_types (List[str]): List of model types to train
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping model names to their metrics
    
    Example:
        >>> results = train_and_evaluate_models(
        ...     X_train, y_train, X_test, y_test,
        ...     model_types=["logistic_regression", "random_forest"]
        ... )
    """
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type}...")
        print('='*50)
        
        model = get_model(model_type)
        
        # Train
        train_metrics = model.train(X_train, y_train)
        print(f"Training metrics: {train_metrics}")
        
        # Evaluate
        eval_metrics = model.evaluate(X_test, y_test)
        print(f"Evaluation metrics: {eval_metrics}")
        
        results[model_type] = {
            **train_metrics,
            **eval_metrics
        }
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Credit Scoring ML Models module loaded")
    print("Available models:")
    print("  - LogisticRegression")
    print("  - RandomForest")
    print("  - GradientBoosting")
    print("\nUsage:")
    print("  model = get_model('logistic_regression')")
    print("  model.train(X_train, y_train)")
    print("  predictions = model.predict(X_test)")
