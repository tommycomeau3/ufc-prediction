"""
scripts/predict_simple.py

Generate predictions using simplified features (no rolling/temporal features).
This ensures compatibility between training and prediction data.

Usage:
    python scripts/predict_simple.py --model models/gbdt_simple.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Project modules
from scripts.preprocess_simple import scale_features_simple
from scripts.features_simple import engineer_features_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming UFC fights with simplified model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/upcoming.csv",
        help="Path to upcoming fights CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gbdt_simple.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions_simple.csv",
        help="Output path for predictions",
    )
    return parser.parse_args()


def load_and_prepare_upcoming_simple(path: str | Path) -> pd.DataFrame:
    """
    Load and prepare upcoming fights data with simplified feature engineering.
    """
    # Read the CSV directly
    df = pd.read_csv(path)
    
    # Add dummy Winner column for preprocessing compatibility
    df["Winner"] = "Red"  # Dummy value, will be ignored
    
    # Apply same preprocessing steps as clean_ufc_data but on DataFrame
    from scripts.preprocess_simple import _coerce_dates, _create_target, _drop_high_missing
    
    df = _coerce_dates(df)
    df = _create_target(df)
    df = _drop_high_missing(df, threshold=0.7)
    df = df.reset_index(drop=True)
    
    # Apply simplified feature engineering (no rolling features)
    df = engineer_features_simple(df)
    
    return df


def predict_upcoming_fights_simple(
    upcoming_df: pd.DataFrame,
    model_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Generate predictions using simplified features.
    """
    # Load the trained model
    print(f"📥 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    try:
        # Use scale_features_simple to get the same preprocessing as training
        X_processed, y_dummy = scale_features_simple(upcoming_df)
        
        print(f"📊 Model expects {model.n_features_in_} features")
        print(f"📊 Upcoming data has {X_processed.shape[1]} features")
        
        # Check for feature mismatch
        if X_processed.shape[1] != model.n_features_in_:
            raise ValueError(f"Feature mismatch: model expects {model.n_features_in_} features, got {X_processed.shape[1]}")
        
        # Make predictions
        y_proba = model.predict_proba(X_processed)[:, 1]
        y_pred = model.predict(X_processed)
        
        print(f"✅ Successfully generated predictions!")
        print(f"📊 Prediction range: {np.min(y_proba):.3f} - {np.max(y_proba):.3f}")
        print(f"📊 Mean prediction: {np.mean(y_proba):.3f}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
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
    
    print("🔮 Generating predictions with simplified model...")
    predictions = predict_upcoming_fights_simple(
        upcoming_df=upcoming_df,
        model_path=args.model,
        output_path=args.output,
    )
    
    print("\n🏆 Top 5 Most Confident Predictions (Simplified Model):")
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