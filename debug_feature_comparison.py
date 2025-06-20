#!/usr/bin/env python3
"""
Debug script to compare features between training and upcoming data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from scripts.preprocess import clean_ufc_data, scale_features
from scripts.features_simple import engineer_features_simple

def compare_features():
    """Compare features between training and upcoming data."""
    
    print("🔍 FEATURE COMPARISON ANALYSIS")
    print("=" * 50)
    
    # Process training data
    print("\n📚 Processing training data...")
    train_df = clean_ufc_data("data/ufc-master.csv")
    train_df = engineer_features_simple(train_df)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Categorical columns in training:")
    cat_cols_train = train_df.select_dtypes(include=["object", "category"]).columns.difference(["Winner"])
    for col in cat_cols_train:
        unique_vals = train_df[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
    
    # Process upcoming data
    print("\n🔮 Processing upcoming data...")
    upcoming_df = pd.read_csv("data/upcoming.csv")
    upcoming_df["Winner"] = "Red"  # Dummy value
    
    from scripts.preprocess import _coerce_dates, _create_target, _drop_high_missing
    upcoming_df = _coerce_dates(upcoming_df)
    upcoming_df = _create_target(upcoming_df)
    upcoming_df = _drop_high_missing(upcoming_df, threshold=0.7)
    upcoming_df = upcoming_df.reset_index(drop=True)
    upcoming_df = engineer_features_simple(upcoming_df)
    
    print(f"Upcoming data shape: {upcoming_df.shape}")
    print(f"Categorical columns in upcoming:")
    cat_cols_upcoming = upcoming_df.select_dtypes(include=["object", "category"]).columns.difference(["Winner"])
    for col in cat_cols_upcoming:
        unique_vals = upcoming_df[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
    
    # Compare categorical values
    print("\n🔍 Categorical value comparison:")
    for col in cat_cols_train:
        if col in cat_cols_upcoming:
            train_vals = set(train_df[col].dropna().unique())
            upcoming_vals = set(upcoming_df[col].dropna().unique())
            
            print(f"\n{col}:")
            print(f"  Training: {len(train_vals)} unique values")
            print(f"  Upcoming: {len(upcoming_vals)} unique values")
            
            if len(train_vals) > 10:  # Only show details for columns with many values
                print(f"  Training sample: {list(train_vals)[:10]}...")
                print(f"  Upcoming sample: {list(upcoming_vals)}")
            else:
                print(f"  Training: {sorted(train_vals)}")
                print(f"  Upcoming: {sorted(upcoming_vals)}")
    
    # Test scaling
    print("\n📐 Testing feature scaling...")
    try:
        X_train, y_train = scale_features(train_df)
        print(f"Training scaled features: {X_train.shape[1]}")
    except Exception as e:
        print(f"Training scaling error: {e}")
    
    try:
        X_upcoming, y_upcoming = scale_features(upcoming_df)
        print(f"Upcoming scaled features: {X_upcoming.shape[1]}")
    except Exception as e:
        print(f"Upcoming scaling error: {e}")

if __name__ == "__main__":
    compare_features()