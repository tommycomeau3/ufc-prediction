"""
scripts/train_simple.py

Train models using simplified features (no rolling/temporal features).
This ensures compatibility between training and prediction data.

Usage:
    python scripts/train_simple.py --model gbdt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Project modules
from scripts.preprocess_simple import clean_ufc_data_simple, scale_features_simple
from scripts.features_simple import engineer_features_simple
from scripts.model import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFC winner prediction model with simplified features")
    parser.add_argument(
        "--data",
        type=str,
        default="data/ufc-master.csv",
        help="Path to historical fight CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gbdt",
        choices=["logreg", "gbdt"],
        help="Which estimator to train",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Custom path to save trained model (defaults depend on --model)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("📥 Loading and cleaning data …")
    df: pd.DataFrame = clean_ufc_data_simple(args.data)
    print(f"✅ Loaded {len(df)} fights")

    print("🛠️  Engineering simplified features (no rolling features) …")
    df = engineer_features_simple(df)
    
    # Count features before scaling
    feature_cols = [c for c in df.columns if c.endswith('_diff') or c.endswith('_streak') or c.endswith('_win_rate') or c.endswith('_ratio') or c.endswith('_flag')]
    print(f"📊 Created {len(feature_cols)} engineered features")

    print("📐 Scaling / encoding (simplified) …")
    X, y = scale_features_simple(df)  # ndarray, Series
    print(f"📊 Final feature count: {X.shape[1]} features for {X.shape[0]} samples")

    print(f"🏋️  Training model ({args.model}) …")
    
    # Set default save paths for simplified models
    if args.save is None:
        if args.model == "gbdt":
            args.save = "models/gbdt_simple.pkl"
        else:
            args.save = "models/logreg_simple.pkl"
    
    model = train_model(
        X=np.asarray(X),
        y=y.values,
        model_type=args.model,
        save_path=args.save,
    )

    print(f"✅ Done. Simplified {args.model} model saved to {args.save}")
    print("🎯 This model should work correctly for upcoming fight predictions!")


if __name__ == "__main__":
    main()