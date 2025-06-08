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
from sklearn.linear_model import LogisticRegression # Imports Logistic Regression model
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report # Import evaluation metrics
from sklearn.model_selection import train_test_split # Imports function to split data


# Defines a helper function to save model
def _save_model(model: Any, path: str | Path = "models/baseline_logreg.pkl") -> None: # Model can be any type and saves model to a string or Path
    """Persist estimator to disk (creates directory if missing)."""
    path = Path(path) # path to Path object
    path.parent.mkdir(parents=True, exist_ok=True) # Ensures parent directory exists (models/)
    joblib.dump(model, path) # Saves model to disk
    print(f"💾 Saved model → {path}")


# --------------------------------------------------------------------------- #
# Public API
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
        penalty="l2", # Add regularization to prevent overfitting
        solver="saga", # Sets algorithm used to optimize weights
        max_iter=500, # Maximum number of iterations for the solver to try to converge
        n_jobs=-1, # How many CPU cores the training can use (-1 means use all)
        class_weight="balanced", # Adjust weight of each class based on frequency
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
    _save_model(model)

    return model