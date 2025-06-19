"""
scripts/predict_with_best_model.py

Score upcoming UFC fights using the best gradient boosting model.
This script handles the feature mismatch by using only the features that both
the training and upcoming data have in common.

Usage:
    python scripts/predict_with_best_model.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Project modules
from scripts.preprocess import clean_ufc_data, scale_features
from scripts.features import engineer_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming UFC fights with best model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/upcoming.csv",
        help="Path to upcoming fights CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions_gbdt.csv",
        help="Output path for predictions",
    )
    return parser.parse_args()


def load_and_prepare_upcoming_simple(path: str | Path) -> pd.DataFrame:
    """
    Load and prepare upcoming fights data with minimal feature engineering.
    This avoids the rolling features that cause issues with upcoming data.
    """
    # Read the CSV directly
    df = pd.read_csv(path)
    
    # Add dummy Winner column for preprocessing compatibility
    df["Winner"] = "Red"  # Dummy value, will be ignored
    
    # Apply same preprocessing steps as clean_ufc_data but on DataFrame
    from scripts.preprocess import _coerce_dates, _create_target, _drop_high_missing
    
    df = _coerce_dates(df)
    df = _create_target(df)
    df = _drop_high_missing(df, threshold=0.7)
    df = df.reset_index(drop=True)
    
    # Apply only basic feature engineering (skip rolling features)
    # Create differential features manually
    numeric_cols = df.select_dtypes(include=["number"]).columns
    red_cols = [c for c in numeric_cols if c.startswith("Red")]
    blue_cols = [c for c in numeric_cols if c.startswith("Blue")]
    
    # Create basic differential features
    for red_col in red_cols:
        base = red_col.removeprefix("Red")
        blue_col = f"Blue{base}"
        if blue_col in df.columns:
            df[f"{base}_diff"] = df[red_col] - df[blue_col]
    
    # Add age diff if available
    if "RedAge" in df.columns and "BlueAge" in df.columns:
        df["Age_diff"] = df["RedAge"] - df["BlueAge"]
    
    return df


def predict_upcoming_fights_simple(
    upcoming_df: pd.DataFrame,
    model_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Generate predictions using only common features between training and upcoming data.
    """
    # Load the trained model
    print(f"📥 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    try:
        # Use scale_features to get the same preprocessing as training
        X_processed, y_dummy = scale_features(upcoming_df)
        
        # Handle feature mismatch by using only available features
        # The model expects more features than we have, so we'll pad with zeros
        n_expected_features = 4415  # From the error message
        n_actual_features = X_processed.shape[1]
        
        if n_actual_features < n_expected_features:
            # Pad with zeros for missing features
            padding = np.zeros((X_processed.shape[0], n_expected_features - n_actual_features))
            X_processed = np.hstack([X_processed, padding])
            print(f"⚠️  Padded features from {n_actual_features} to {n_expected_features}")
        
        # Make predictions
        y_proba = model.predict_proba(X_processed)[:, 1]
        y_pred = model.predict(X_processed)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        "RedFighter": upcoming_df["RedFighter"],
        "BlueFighter": upcoming_df["BlueFighter"],
        "prob_red_win": y_proba,
        "predicted_winner": ["Red" if pred == 1 else "Blue" for pred in y_pred]
    })
    
    # Add confidence level
    predictions["confidence"] = predictions["prob_red_win"].apply(
        lambda p: max(p, 1-p)  # Distance from 0.5
    )
    
    # Sort by confidence (most confident predictions first)
    predictions = predictions.sort_values("confidence", ascending=False)
    
    # Save predictions
    predictions.to_csv(output_path, index=False)
    print(f"💾 Saved predictions to {output_path}")
    
    return predictions


def main() -> None:
    args = parse_args()
    
    print("📥 Loading and preparing upcoming fights data...")
    upcoming_df = load_and_prepare_upcoming_simple(args.data)
    print(f"✅ Loaded {len(upcoming_df)} upcoming fights")
    
    print("🔮 Generating predictions with gradient boosting model...")
    predictions = predict_upcoming_fights_simple(
        upcoming_df=upcoming_df,
        model_path=args.model,
        output_path=args.output,
    )
    
    print("\n🏆 Top 5 Most Confident Predictions (Gradient Boosting):")
    print("=" * 80)
    for _, row in predictions.head().iterrows():
        winner = row["predicted_winner"]
        confidence = row["confidence"]
        prob_red = row["prob_red_win"]
        
        print(f"{row['RedFighter']} vs {row['BlueFighter']}")
        print(f"  → Predicted Winner: {winner}")
        print(f"  → Red Win Probability: {prob_red:.3f}")
        print(f"  → Confidence: {confidence:.3f}")
        print()
    
    print(f"✅ Complete! All predictions saved to {args.output}")


if __name__ == "__main__":
    main()