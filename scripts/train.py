"""
scripts/train.py

Command-line training orchestrator for the UFC fight-outcome project.

Example
-------
python scripts/train.py --model gbdt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Project modules
from scripts.preprocess import clean_ufc_data, scale_features
from scripts.features import engineer_features
from scripts.model import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFC winner prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/ufc-master.csv",
        help="Path to historical fight CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
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
    df: pd.DataFrame = clean_ufc_data(args.data)

    print("🛠️  Engineering features …")
    df = engineer_features(df)

    print("📐 Scaling / encoding …")
    X, y = scale_features(df)  # ndarray, Series

    print(f"🏋️  Training model ({args.model}) …")
    model = train_model(
        X=np.asarray(X),
        y=y.values,
        model_type=args.model,
        save_path=args.save,
    )

    print("✅ Done. Model persisted; ready for inference.")


if __name__ == "__main__":
    main()