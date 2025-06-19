"""
scripts/predict_upcoming.py

Score upcoming UFC fights using the best trained model.

Usage:
    python scripts/predict_upcoming.py
    python scripts/predict_upcoming.py --data data/upcoming.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

# Project modules
from scripts.preprocess import clean_ufc_data
from scripts.features import engineer_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming UFC fight outcomes")
    parser.add_argument(
        "--data",
        type=str,
        default="data/upcoming.csv",
        help="Path to upcoming fights CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/stage1_logreg_pipeline.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output path for predictions",
    )
    return parser.parse_args()


def load_and_prepare_upcoming(path: str | Path) -> pd.DataFrame:
    """
    Load and prepare upcoming fights data for prediction.
    
    Note: Upcoming fights don't have a Winner column, so we'll add a dummy one
    for the preprocessing pipeline to work.
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
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Add missing columns that the pipeline expects (with default values)
    # Do this AFTER feature engineering so they don't get dropped
    missing_cols = {
        'TotalFightTimeSecs': 900,  # Default 15 minutes
        'Finish': 'No',  # Default no finish
        'FinishDetails': '',  # Empty string
        'FinishRound': 3,  # Default to final round
        'FinishRoundTime': '5:00',  # Default to end of round
        'EmptyArena': False  # Default not empty arena
    }
    
    for col, default_val in missing_cols.items():
        if col not in df.columns:
            df[col] = default_val
    
    return df


def predict_upcoming_fights(
    upcoming_df: pd.DataFrame,
    model_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Generate predictions for upcoming fights.
    
    Parameters
    ----------
    upcoming_df : pd.DataFrame
        Preprocessed upcoming fights data
    model_path : str | Path
        Path to trained model
    output_path : str | Path
        Where to save predictions
        
    Returns
    -------
    pd.DataFrame
        Predictions with columns: RedFighter, BlueFighter, prob_red_win, predicted_winner
    """
    # Load the trained model
    print(f"📥 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # The pipeline model handles preprocessing internally
    try:
        # Prepare features (drop target column)
        X = upcoming_df.drop(columns=["target"])
        
        # Make predictions using the pipeline
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"Model type: {type(model)}")
        print(f"X shape: {X.shape}")
        print(f"X columns: {list(X.columns)}")
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
    upcoming_df = load_and_prepare_upcoming(args.data)
    print(f"✅ Loaded {len(upcoming_df)} upcoming fights")
    
    print("🔮 Generating predictions...")
    predictions = predict_upcoming_fights(
        upcoming_df=upcoming_df,
        model_path=args.model,
        output_path=args.output,
    )
    
    print("\n🏆 Top 5 Most Confident Predictions:")
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