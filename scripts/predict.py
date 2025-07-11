#!/usr/bin/env python3
"""
scripts/predict.py

Generate predictions for upcoming fights using the trained model pipeline.

Usage:
    python scripts/predict.py
"""

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import sys

# Ensure project root is on sys.path when script executed via `python scripts/predict.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Project feature utilities
from scripts.features import engineer_features

# Paths
MODEL_PATH = Path("models/stage1_logreg_pipeline.pkl")
UPCOMING_CSV = Path("data/upcoming.csv")
OUT_CSV = Path("data/upcoming_with_preds.csv")


def main() -> None:
    # 1. Load model pipeline -------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    # 2. Read upcoming fights --------------------------------------------------
    if not UPCOMING_CSV.exists():
        raise FileNotFoundError(f"Upcoming fights CSV not found: {UPCOMING_CSV}")
    upcoming_df = pd.read_csv(UPCOMING_CSV)
    if upcoming_df.empty:
        raise ValueError("Upcoming fights CSV is empty")

    # 3. Basic date parsing (needed for time-aware features) --------------------
    if "Date" in upcoming_df.columns:
        upcoming_df["Date"] = pd.to_datetime(upcoming_df["Date"], errors="coerce")

    # 4. Feature engineering (add *_diff, streaks, etc.) ------------------------
    upcoming_df = engineer_features(upcoming_df)

    # 5. Align columns with training schema ------------------------------------
    if hasattr(pipeline, "feature_names_in_"):
        required_cols = list(pipeline.feature_names_in_)
    elif hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        required_cols = list(pipeline.named_steps["preprocessor"].feature_names_in_)
    else:
        required_cols = list(upcoming_df.columns)  # fallback

    # Add any missing columns as NaN so ColumnTransformer can handle them
    missing_cols = set(required_cols) - set(upcoming_df.columns)
    for col in missing_cols:
        upcoming_df[col] = np.nan

    # Preserve column order expected by the preprocessor
    upcoming_df = upcoming_df[required_cols]

    # 6. Generate predictions ---------------------------------------------------
    prob_f1_win = pipeline.predict_proba(upcoming_df)[:, 1]   # P(win) for fighter_1
    pred_f1_win = (prob_f1_win >= 0.5).astype(int)            # hard label

    # 7. Save / inspect results -----------------------------------------------
    upcoming_df["prob_f1_win"] = prob_f1_win
    upcoming_df["pred_f1_win"] = pred_f1_win
    upcoming_df.to_csv(OUT_CSV, index=False)
    
    # Display predictions in terminal
    print("\n" + "="*80)
    print("FIGHT PREDICTIONS")
    print("="*80)
    
    for i, row in upcoming_df.iterrows():
        red_fighter = row['RedFighter']
        blue_fighter = row['BlueFighter']
        prob_red_win = row['prob_f1_win']
        prob_blue_win = 1 - prob_red_win
        predicted_winner = red_fighter if row['pred_f1_win'] == 1 else blue_fighter
        confidence = max(prob_red_win, prob_blue_win)
        
        print(f"\n{red_fighter} vs {blue_fighter}")
        print(f"  {red_fighter}: {prob_red_win:.1%} chance to win")
        print(f"  {blue_fighter}: {prob_blue_win:.1%} chance to win")
        print(f"  → PREDICTION: {predicted_winner} wins ({confidence:.1%} confidence)")
        
        # Add date and location if available
        if 'Date' in row and pd.notna(row['Date']):
            print(f"  📅 {row['Date'].strftime('%B %d, %Y') if hasattr(row['Date'], 'strftime') else row['Date']}")
        if 'Location' in row and pd.notna(row['Location']):
            print(f"  📍 {row['Location']}")
    
    print("\n" + "="*80)
    print(f"Predictions also saved to: {OUT_CSV.resolve()}")
    print("="*80)


if __name__ == "__main__":
    main()