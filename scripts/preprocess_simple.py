"""
scripts/preprocess_simple.py

Simplified preprocessing that avoids categorical explosion for fighter names and locations.
This ensures compatibility between training and prediction data.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV with common encodings fallback."""
    encodings_to_try: List[str] = ["utf-8", "latin1"]
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            warnings.warn(f"Failed to read with encoding={enc}. Retrying…")
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date column to pandas datetime (if present)."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target column: 1 if Red wins, 0 if Blue wins."""
    if "Winner" not in df.columns:
        raise ValueError("Column 'Winner' missing in dataset.")
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["target"] = (df["Winner"] == "Red").astype(int)
    return df


def _drop_high_missing(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Drop columns whose missing-value ratio exceeds threshold."""
    keep_cols = df.columns[df.isna().mean() < threshold]
    return df[keep_cols]


def clean_ufc_data_simple(path: str | Path) -> pd.DataFrame:
    """Load and clean the UFC dataset with simplified approach."""
    df = _read_csv(path)
    df = _coerce_dates(df)
    df = _create_target(df)
    df = _drop_high_missing(df, threshold=0.7)
    df = df.reset_index(drop=True)
    return df


def scale_features_simple(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Prepare design-matrix X and label vector y with simplified categorical handling.
    
    This version excludes high-cardinality categorical features like fighter names
    and locations that cause feature explosion.
    """
    if "target" not in df.columns:
        raise KeyError("DataFrame must contain 'target' column.")

    y = df["target"]

    # Separate features
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop("target")
    
    # Only include low-cardinality categorical columns
    all_categorical_cols = (
        df.select_dtypes(include=["object", "category"])
        .columns.difference(["Winner"])  # drop raw winner label
    )
    
    # Filter out high-cardinality categorical columns
    low_cardinality_categorical_cols = []
    for col in all_categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 20:  # Only include columns with ≤20 unique values
            low_cardinality_categorical_cols.append(col)
        else:
            print(f"⚠️  Excluding high-cardinality column: {col} ({unique_count} unique values)")
    
    categorical_cols = low_cardinality_categorical_cols

    # Pipelines
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    X_scaled = preprocessor.fit_transform(df)
    return X_scaled, y