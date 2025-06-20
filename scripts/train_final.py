"""
scripts/train_final.py

Final training script that saves both the model and preprocessor for exact feature alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Project modules
from scripts.preprocess_simple import clean_ufc_data_simple
from scripts.features_simple import engineer_features_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFC model with saved preprocessor")
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
    return parser.parse_args()


def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create and return the preprocessor pipeline."""
    # Separate features
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop("target")
    
    # Only include low-cardinality categorical columns
    all_categorical_cols = (
        df.select_dtypes(include=["object", "category"])
        .columns.difference(["Winner"])
    )
    
    categorical_cols = []
    for col in all_categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 20:
            categorical_cols.append(col)
        else:
            print(f"⚠️  Excluding high-cardinality column: {col} ({unique_count} unique values)")

    # Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    
    return preprocessor


def main() -> None:
    args = parse_args()

    print("📥 Loading and cleaning data …")
    df = clean_ufc_data_simple(args.data)
    print(f"✅ Loaded {len(df)} fights")

    print("🛠️  Engineering simplified features …")
    df = engineer_features_simple(df)

    print("📐 Creating preprocessor …")
    preprocessor = create_preprocessor(df)
    
    # Prepare data
    y = df["target"]
    X_processed = preprocessor.fit_transform(df)
    
    print(f"📊 Final feature count: {X_processed.shape[1]} features for {X_processed.shape[0]} samples")

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🏋️  Training {args.model} model …")
    
    # Create model
    if args.model == "gbdt":
        model = GradientBoostingClassifier(random_state=42)
        model_path = "models/gbdt_final.pkl"
        preprocessor_path = "models/gbdt_final_preprocessor.pkl"
    else:
        model = LogisticRegression(
            penalty="l2", solver="saga", max_iter=10000, 
            tol=1e-4, n_jobs=-1, class_weight="balanced"
        )
        model_path = "models/logreg_final.pkl"
        preprocessor_path = "models/logreg_final_preprocessor.pkl"
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_proba)
    
    print("✅ Validation metrics:")
    print(f"   • Accuracy : {acc:0.4f}")
    print(f"   • ROC-AUC  : {roc:0.4f}")
    print("   • Class report\n", classification_report(y_val, y_pred, digits=3))
    
    # Retrain on full data
    model.fit(X_processed, y)
    
    # Save model and preprocessor
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"💾 Saved model to {model_path}")
    print(f"💾 Saved preprocessor to {preprocessor_path}")
    print("✅ Done! Model and preprocessor saved for consistent predictions.")


if __name__ == "__main__":
    main()