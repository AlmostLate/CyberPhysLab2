"""
Data loader for Adult Income dataset.

This module provides utilities for loading and preprocessing
the UCI Adult Income dataset for credit scoring analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DatasetConfig, DATA_DIR


# Column names for Adult dataset
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]


def download_dataset(url: str = DatasetConfig.URL, save_path: str = DatasetConfig.PATH) -> pd.DataFrame:
    """
    Download the Adult Income dataset from UCI repository.
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the dataset
    
    Returns:
        pd.DataFrame: Downloaded dataset
    
    Raises:
        Exception: If download fails
    """
    print(f"Downloading dataset from {url}...")
    
    # Create data directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download using pandas
        df = pd.read_csv(
            url,
            names=ADULT_COLUMNS,
            skiprows=1,  # Skip header row if present
            na_values="?",  # Handle missing values
            sep=", ",
            engine="python"
        )
        
        # Save locally
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
        
        return df
    
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def load_adult_dataset(
    force_download: bool = False,
    local_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load the Adult Income dataset.
    
    Attempts to load from local path first, then downloads if necessary.
    
    Args:
        force_download (bool): Force re-download even if local file exists
        local_path (str, optional): Custom local path
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Example:
        >>> df = load_adult_dataset()
        >>> print(f"Loaded {len(df)} records")
    """
    path = local_path or DatasetConfig.PATH
    
    # Check if file exists
    if Path(path).exists() and not force_download:
        print(f"Loading dataset from {path}")
        df = pd.read_csv(path)
    else:
        # Download
        try:
            df = download_dataset(save_path=path)
        except:
            # Fallback: try to use sample data if download fails
            print("Download failed, using inline sample data for demonstration...")
            df = _get_sample_data()
    
    # Basic cleaning
    df = _clean_dataset(df)
    
    print(f"Loaded dataset with {len(df)} records and {len(df.columns)} columns")
    
    return df


def _get_sample_data() -> pd.DataFrame:
    """Get sample data for demonstration when download fails."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "age": np.random.randint(18, 65, n_samples),
        "workclass": np.random.choice(["Private", "Self-emp", "Government"], n_samples),
        "fnlwgt": np.random.randint(10000, 500000, n_samples),
        "education": np.random.choice(["Bachelors", "HS-grad", "Masters", "Some-college"], n_samples),
        "education_num": np.random.randint(1, 16, n_samples),
        "marital_status": np.random.choice(["Married", "Single", "Divorced"], n_samples),
        "occupation": np.random.choice(["Tech", "Sales", "Service", "Manual"], n_samples),
        "relationship": np.random.choice(["Husband", "Wife", "Own-child", "Unmarried"], n_samples),
        "race": np.random.choice(["White", "Black", "Asian", "Other"], n_samples),
        "sex": np.random.choice(["Male", "Female"], n_samples),
        "capital_gain": np.random.exponential(1000, n_samples).astype(int),
        "capital_loss": np.random.exponential(100, n_samples).astype(int),
        "hours_per_week": np.random.randint(20, 60, n_samples),
        "native_country": np.random.choice(["United-States", "Other"], n_samples),
        "income": np.random.choice([" <=50K", " >50K"], n_samples, p=[0.75, 0.25])
    }
    
    return pd.DataFrame(data)


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip() if hasattr(df[col], "str") else df[col]
    
    # Handle missing values
    # For simplicity, drop rows with missing values
    # In production, you might want to impute
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df.reset_index(drop=True)


def preprocess_dataset(
    df: pd.DataFrame,
    encode_target: bool = True,
    scale_numerical: bool = True
) -> Tuple[pd.DataFrame, Optional[LabelEncoder], Optional[StandardScaler]]:
    """
    Preprocess the Adult Income dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        encode_target (bool): Whether to encode target variable
        scale_numerical (bool): Whether to scale numerical features
    
    Returns:
        Tuple containing:
            - Preprocessed DataFrame
            - Label encoder for target (if encode_target=True)
            - StandardScaler for numerical features (if scale_numerical=True)
    
    Example:
        >>> df = load_adult_dataset()
        >>> df_processed, le, scaler = preprocess_dataset(df)
    """
    df = df.copy()
    
    # Store target encoder if needed
    target_encoder = None
    if encode_target:
        target_encoder = LabelEncoder()
        df["income_encoded"] = target_encoder.fit_transform(df["income"])
        print(f"Target classes: {target_encoder.classes_}")
    
    # Encode categorical features
    categorical_cols = DatasetConfig.CATEGORICAL_COLUMNS
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Scale numerical features
    scaler = None
    if scale_numerical:
        scaler = StandardScaler()
        numerical_cols = DatasetConfig.NUMERICAL_COLUMNS
        
        # Select only columns that exist
        existing_numerical = [c for c in numerical_cols if c in df.columns]
        
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[existing_numerical]),
            columns=existing_numerical,
            index=df.index
        )
        
        # Replace original numerical columns with scaled versions
        for col in existing_numerical:
            df[f"{col}_scaled"] = df_scaled[col]
    
    return df, target_encoder, scaler


def prepare_train_test_split(
    df: pd.DataFrame,
    target_column: str = "income",
    test_size: float = 0.2,
    random_state: int = 42,
    use_encoded: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train and test splits.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed
        use_encoded (bool): Use encoded target if available
    
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    
    Example:
        >>> df = load_adult_dataset()
        >>> df_proc = preprocess_dataset(df)
        >>> X_train, X_test, y_train, y_test = prepare_train_test_split(df_proc)
    """
    # Determine which columns to use as features
    exclude_cols = ["income", "income_encoded"] + DatasetConfig.CATEGORICAL_COLUMNS
    
    # Include scaled/encoded numerical columns
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            # Include encoded categorical and scaled numerical
            if "_encoded" in col or "_scaled" in col or col in DatasetConfig.NUMERICAL_COLUMNS:
                feature_cols.append(col)
    
    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    
    # Prepare X and y
    X = df[feature_cols]
    
    # Prepare y
    if use_encoded and "income_encoded" in df.columns:
        y = df["income_encoded"]
    else:
        y = df["income"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_encoded else None
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """
    Get feature column names for model training.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
    
    Returns:
        List[str]: Feature column names
    """
    exclude = ["income", "income_encoded"] + DatasetConfig.CATEGORICAL_COLUMNS
    
    features = []
    for col in df.columns:
        if col not in exclude:
            if "_encoded" in col or "_scaled" in col or col in DatasetConfig.NUMERICAL_COLUMNS:
                features.append(col)
    
    return list(dict.fromkeys(features))


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset
    
    Returns:
        Dict with dataset statistics
    """
    info = {
        "num_records": len(df),
        "num_features": len(df.columns),
        "num_classes": df["income"].nunique() if "income" in df.columns else 0,
        "class_distribution": df["income"].value_counts().to_dict() if "income" in df.columns else {},
        "missing_values": df.isnull().sum().to_dict(),
        "numerical_columns": DatasetConfig.NUMERICAL_COLUMNS,
        "categorical_columns": DatasetConfig.CATEGORICAL_COLUMNS
    }
    
    return info


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Testing data loader...")
    
    # Load dataset
    df = load_adult_dataset()
    
    print("\nDataset Info:")
    info = get_dataset_info(df)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nClass distribution:")
    print(df["income"].value_counts())
