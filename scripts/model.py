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

from __future__ import annotations # Allows to use class/type names bdefore definition

from pathlib import Path # Bring in Path class from pathlib module
from typing import Tuple, Any # Lets you specify fixed-length tuples and disable type checking

import joblib # Allows for saving models
import numpy as np # Numerical operations
from sklearn.linear_model import LogisticRegression  # Imports Logistic Regression model
from sklearn.ensemble import GradientBoostingClassifier  # Tree-based baseline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report # Import evaluation metrics
from sklearn.model_selection import train_test_split # Imports function to split data


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _save_model(model: Any, path: str | Path) -> None:
    """Persist estimator to disk (creates directory if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Saved model → {path}")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Return accuracy and ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)
    print("✅ Validation metrics:")
    print(f"   • Accuracy : {acc:0.4f}")
    print(f"   • ROC-AUC  : {roc:0.4f}")
    print("   • Class report\n", classification_report(y_true, y_pred, digits=3))
    return acc, roc


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str | Path | None = None,
):
    """
    Generic training wrapper supporting multiple estimators.

    Parameters
    ----------
    X, y : np.ndarray
        Feature matrix and binary target vector.
    model_type : {"logreg", "gbdt"}
        Which estimator to train.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_type == "logreg":
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=10000,  # further increased to avoid ConvergenceWarning
            tol=1e-4,
            n_jobs=-1,
            class_weight="balanced",
        )
        default_path = "models/baseline_logreg.pkl"
    elif model_type == "gbdt":
        model = GradientBoostingClassifier(random_state=random_state)
        default_path = "models/gbdt.pkl"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # -------------------- Fit & evaluate -------------------- #
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    _, val_roc = _evaluate(y_val, y_pred, y_proba)

    # ---------------- Compare against existing best ---------------- #
    best_path = Path("models/best.pkl")
    best_roc = -1.0
    if best_path.exists():
        try:
            best_model = joblib.load(best_path)
            best_pred = best_model.predict(X_val)
            best_proba = best_model.predict_proba(X_val)[:, 1]
            best_roc = roc_auc_score(y_val, best_proba)
            print(f"📊 Current best ROC-AUC = {best_roc:0.4f}")
        except Exception as err:
            print(f"⚠️  Failed to evaluate existing best model: {err}")

    if val_roc >= best_roc:
        print(f"🏆 New model beats / matches best (ROC {val_roc:0.4f} ≥ {best_roc:0.4f}). Saving as best.pkl")
        _save_model(model, best_path)
    else:
        print(f"ℹ️  New model ROC {val_roc:0.4f} < best {best_roc:0.4f}. best.pkl remains unchanged.")

    # --------------------- Refit on full -------------------- #
    model.fit(X, y)

    # --------------------- Persist (model-type specific) ---- #
    out_path = save_path or default_path
    _save_model(model, out_path)
    return model


# --------------------------------------------------------------------------- #
# Backwards-compat convenience
# --------------------------------------------------------------------------- #
def train_and_evaluate_model(
    X: np.ndarray, # NumPy array of shape (n_samples, n_features)
    y: np.ndarray, # NumPy array of shape (n_samples,)
    test_size: float = 0.2, # 20% for validation
    random_state: int = 42, # Same split everytime
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
    # Splits data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Creates logistic regression model
    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=10000,  # further increased to avoid ConvergenceWarning
        tol=1e-4,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Fits model based on training data
    model.fit(X_train, y_train)

    # -------------------- Evaluation -------------------- #
    y_pred = model.predict(X_val) # Uses trained model to predict class labels (0 or 1) on validation set
    y_proba = model.predict_proba(X_val)[:, 1] # Predicts class probabilities for each sample in X_val

    acc = accuracy_score(y_val, y_pred) # Computes fraction of correct predictions
    roc = roc_auc_score(y_val, y_proba)  # Measures the model's ability to rank positive vs. negative cases (Receiver Operating Characteristic - Area Under Curve)

    print("✅ Validation metrics:")
    print(f"   • Accuracy : {acc:0.4f}")
    print(f"   • ROC-AUC  : {roc:0.4f}")
    print("   • Class report\n", classification_report(y_val, y_pred, digits=3))

    # Fit on full data before returning
    model.fit(X, y)
    # Persist model to default location
    _save_model(model, "models/baseline_logreg.pkl")
    
    return model