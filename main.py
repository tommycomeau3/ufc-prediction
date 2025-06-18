"""
main.py

Stage-1 baseline:
- End-to-end Pipeline (preprocessing + model) using 5-fold cross-validated
  logistic regression.
- 15% stratified hold-out test set for an unbiased estimate.
- Reports mean and std CV AUC plus test metrics, then persists the fitted
  Pipeline for inference.

Run:
    python main.py
"""

from __future__ import annotations
from pathlib import Path
import joblib
from joblib import Memory
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project modules
from scripts.features import engineer_features
from scripts.preprocess import (
    _read_csv,
    _coerce_dates,
    _create_target,
    _drop_high_missing,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_and_prepare(path: str | Path) -> pd.DataFrame:
    """Load CSV and apply basic cleaning."""
    df = _read_csv(path)
    df = _coerce_dates(df)
    df = _create_target(df)
    df = _drop_high_missing(df, threshold=0.7).reset_index(drop=True)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Return a ColumnTransformer that matches *df*'s schema."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop("target")
    categorical_cols = (
        df.select_dtypes(include=["object", "category"]).columns.difference(["Winner"])
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # keep sparse compatibility
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # default sparse_output=True keeps the matrix sparse
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #


def main() -> None:
    # 1) Load + clean
    df_raw = load_and_prepare("data/ufc-master.csv")
    print("✅ Cleaned shape:", df_raw.shape)

    # 2) Feature engineering
    df_feat = engineer_features(df_raw, drop_original=False)
    print("✅ After engineering shape:", df_feat.shape)

    # 3) Train / test split (15 % hold-out)
    y = df_feat["target"]
    X = df_feat.drop(columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print("✅ Train:", X_train.shape, "| Test:", X_test.shape)

    # 4) Build Pipeline
    preprocessor = build_preprocessor(df_feat)
    # cache the preprocessing step to avoid recomputing for each C value
    memory = Memory(location=".cache", verbose=0)

    # Increase max_iter and keep a strict tolerance to avoid ConvergenceWarning
    clf = LogisticRegressionCV(
        Cs=5,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        solver="saga",
        penalty="l2",
        max_iter=10000,   # was 2000
        tol=1e-4,         # explicit (same as default but shown for clarity)
        class_weight="balanced",
        n_jobs=-1,
        refit=True,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ],
        memory=memory,
    )

    # 5) Fit Pipeline (CV happens internally)
    pipe.fit(X_train, y_train)
    cv_scores = pipe.named_steps["clf"].scores_[1]
    cv_auc_mean = cv_scores.mean()
    cv_auc_std = cv_scores.std()
    print(f"✅ 5-fold CV ROC-AUC : {cv_auc_mean:0.4f} ± {cv_auc_std:0.4f}")

    # 6) Evaluate on hold-out test set
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    print("✅ Test metrics:")
    print(f"   - Accuracy : {test_acc:0.4f}")
    print(f"   - ROC-AUC  : {test_auc:0.4f}")
    print("   - Class report\n", classification_report(y_test, y_pred, digits=3))

    # 7) Refit on full data (train+test) for production
    pipe.fit(X, y)
    out_path = Path("models/stage1_logreg_pipeline.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"💾 Saved Pipeline → {out_path}")


if __name__ == "__main__":
    main()
