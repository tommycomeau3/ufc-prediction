"""
scripts/model.py

Baseline modelling utilities for UFC fight outcome prediction.

This module currently implements:
    • train_and_evaluate_model: simple train/valid split +
      LogisticRegression baseline with standard metrics.

Future work (see docs/architecture.md):
    • Add tree ensembles (RandomForest, XGBoost, LightGBM, CatBoost)
    • Hyper-parameter tuning via Optuna / GridSearchCV
    • Feature-importance & SHAP
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


def _save_model(model: Any, path: str | Path = "models/baseline_logreg.pkl") -> None:
    """Persist estimator to disk (creates directory if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Saved model → {path}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a baseline Logistic Regression classifier and print evaluation metrics.

    Parameters
    ----------
    X, y : np.ndarray
        Feature matrix and binary target vector.
    test_size : float, optional
        Fraction of data reserved for evaluation (default 0.2).
    random_state : int, optional
        Reproducibility seed (default 42).

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Trained estimator fitted on **all** data (train + eval) for downstream use.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # -------------------- Evaluation -------------------- #
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_proba)

    print("✅ Validation metrics:")
    print(f"   • Accuracy : {acc:0.4f}")
    print(f"   • ROC-AUC  : {roc:0.4f}")
    print("   • Class report\n", classification_report(y_val, y_pred, digits=3))

    # Fit on full data before returning
    model.fit(X, y)
    _save_model(model)

    return model