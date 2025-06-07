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
def _read_csv(path: str | Path) -> pd.DataFrame: # Takes a path and returns a DataFrame
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
    encodings_to_try: List[str] = ["utf-8", "latin1"] # Options of encodings to try
    for enc in encodings_to_try: # Trys utf-8 first then retrys with latin1
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            warnings.warn(f"Failed to read with encoding={enc}. Retrying…")
    # Final attempt: let pandas infer
    return pd.read_csv(path, encoding="utf-8", errors="replace") # Falls back to utf-8 and replaces unreadable chararacters


# --------------------------------------------------------------------------- #
# 2. Cleaning helpers
# --------------------------------------------------------------------------- #
def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date column to pandas datetime (if present)."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce") # Convert text-based date strings into proper datetime64 format.

    return df # Returns DataFrame


def _create_target(df: pd.DataFrame) -> pd.DataFrame: # Checks if Date is in DataFrame
    """
    Add binary `target` column: 1 if Red wins, 0 if Blue wins.
    Removes rows without a definitive winner.
    """
    if "Winner" not in df.columns: # Checks if "Winner" is present
        raise ValueError("Column 'Winner' missing in dataset.") # Raises error if no "Winner" column
    df = df[df["Winner"].isin(["Red", "Blue"])].copy() # Keeps rows where Winner is Red or Blue
    df["target"] = (df["Winner"] == "Red").astype(int) # Compares each row's "Winner" to "Red" (Red = 1/Blue = 0)
    return df


def _drop_high_missing(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame: # threshold=0.7 means: "drop columns with >70% missing values
    """
    Drop columns whose missing-value ratio exceeds `threshold`.
    """
    keep_cols = df.columns[df.isna().mean() < threshold] # df.isna().mean() calculates the fraction of missing values for each column
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


def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]: # Takes a DataFrame and returns a Tuple of a Numpy array of features and pandas Series of Target lables
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
    if "target" not in df.columns: # Checks for target column
        raise KeyError("DataFrame must contain 'target' column. Have you run clean_ufc_data()?")

    y = df["target"] # Sets target column to y

    # Separate features
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop("target") # Selects all numeric columns (except target)
    # Select all category type columns
    categorical_cols = (
        df.select_dtypes(include=["object", "category"])
        .columns.difference(["Winner"])  # drop raw winner label
    )

    # A Pipeline is a tool from scikit-learn that lets you chain together multiple preprocessing steps
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), # For each column finds median and replaces any missing value with median
            ("scaler", StandardScaler()), # Scale the values
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # Finds most common value for each Column and puts into NaN
            # scikit-learn ≥1.2 renamed `sparse` → `sparse_output`
            # handle_unknown="ignore" - if you try to encode a new category at prediction time, don’t crash — just ignore it.
            # sparse_output=False - dense NumPy array
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # Converts categories to binary columns
        ]
    )

    preprocessor: ColumnTransformer = ColumnTransformer( # Declares a variable named `preprocessor` of type `ColumnTransformer`
        transformers=[
            ("num", numeric_pipeline, numeric_cols), # Applies your `numeric_pipeline` to all `numeric_cols`
            ("cat", categorical_pipeline, categorical_cols), # Applies your categorical_pipeline to all categorical_cols
        ],
        remainder="drop", # Any columns not listed in numeric_cols or categorical_cols will be dropped
    )

    X_scaled = preprocessor.fit_transform(df) # Applies preprocessing pipepline to DataFrame
    return X_scaled, y