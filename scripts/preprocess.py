"""
scripts/preprocess.py

Data loading, cleaning, and feature-preparation helpers for the UFC fight
outcome prediction pipeline.
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


# --------------------------------------------------------------------------- #
# 1. Data loading & initial sanity checks
# --------------------------------------------------------------------------- #
def _read_csv(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV with common encodings fallback.

    Parameters
    ----------
    path : str | pathlib.Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    encodings_to_try: List[str] = ["utf-8", "latin1"]
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            warnings.warn(f"Failed to read with encoding={enc}. Retrying…")
    # Final attempt: let pandas infer
    return pd.read_csv(path, encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# 2. Cleaning helpers
# --------------------------------------------------------------------------- #
def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date column to pandas datetime (if present)."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary `target` column: 1 if Red wins, 0 if Blue wins.
    Removes rows without a definitive winner.
    """
    if "Winner" not in df.columns:
        raise ValueError("Column 'Winner' missing in dataset.")
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["target"] = (df["Winner"] == "Red").astype(int)
    return df


def _drop_high_missing(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Drop columns whose missing-value ratio exceeds `threshold`.
    """
    keep_cols = df.columns[df.isna().mean() < threshold]
    return df[keep_cols]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def clean_ufc_data(path: str | Path) -> pd.DataFrame:
    """
    Load and clean the UFC dataset.

    Steps
    -----
    1. Read CSV with tolerant encodings.
    2. Parse date.
    3. Create binary `target`.
    4. Drop columns with >70 % missingness.
    5. Reset index.

    Returns
    -------
    Cleaned `pd.DataFrame`.
    """
    df = _read_csv(path)
    df = _coerce_dates(df)
    df = _create_target(df)
    df = _drop_high_missing(df, threshold=0.7)
    df = df.reset_index(drop=True)
    return df


def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Prepare design-matrix `X` and label vector `y`.

    • Numeric columns → median imputation + StandardScaler  
    • Categorical columns (object/category) → most-frequent imputation + OneHot

    Returns
    -------
    X_scaled : np.ndarray
        Scaled/encoded feature matrix.
    y        : pd.Series
        Binary target vector (1 = Red wins).
    """
    if "target" not in df.columns:
        raise KeyError("DataFrame must contain 'target' column. Have you run clean_ufc_data()?")

    y = df["target"]

    # Separate features
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop("target")
    categorical_cols = (
        df.select_dtypes(include=["object", "category"])
        .columns.difference(["Winner"])  # drop raw winner label
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # scikit-learn ≥1.2 renamed `sparse` → `sparse_output`
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