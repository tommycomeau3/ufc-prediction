"""
scripts/predict_final.py

Final prediction script that uses the saved preprocessor for exact feature alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Project modules
from scripts.preprocess_simple import _coerce_dates, _create_target, _drop_high_missing
from scripts.features_simple import engineer_features_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with saved model and preprocessor")
    parser.add_argument(
        "--data",
        type=str,
        default="data/upcoming.csv",
        help="Path to upcoming fights CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gbdt_final.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="models/gbdt_final_preprocessor.pkl",
        help="Path to fitted preprocessor",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions_final.csv",
        help="Output path for predictions",
    )
    return parser.parse_args()


def load_and_prepare_upcoming(path: str | Path) -> pd.DataFrame:
    """Load and prepare upcoming fights data."""
    df = pd.read_csv(path)
    
    print(f"📊 Original upcoming data columns: {len(df.columns)}")
    
    # Add dummy Winner column for preprocessing compatibility
    df["Winner"] = "Red"
    
    # Add missing columns that training data expects (check the CSV - some might already exist)
    missing_columns = {
        'Finish': 'No',
        'FinishDetails': '',
        'FinishRound': 3,
        'FinishRoundTime': '5:00',
        'TotalFightTimeSecs': 900,  # 15 minutes default
    }
    
    for col, default_val in missing_columns.items():
        if col not in df.columns:
            df[col] = default_val
            print(f"➕ Added missing column: {col} = {default_val}")
        else:
            print(f"✅ Column already exists: {col}")
    
    # Apply same preprocessing steps
    df = _coerce_dates(df)
    df = _create_target(df)
    
    print(f"📊 After initial preprocessing: {len(df.columns)} columns")
    print(f"📊 Sample columns: {list(df.columns)[:10]}...")
    
    # Store important columns before dropping high missing
    important_cols = ['Finish', 'FinishRound', 'TotalFightTimeSecs', 'EmptyArena']
    preserved_data = {}
    for col in important_cols:
        if col in df.columns:
            preserved_data[col] = df[col].copy()
    
    df = _drop_high_missing(df, threshold=0.7)
    print(f"📊 After dropping high missing: {len(df.columns)} columns")
    
    # Restore important columns that may have been dropped
    for col, data in preserved_data.items():
        if col not in df.columns:
            df[col] = data
            print(f"🔄 Restored dropped column: {col}")
    
    df = df.reset_index(drop=True)
    
    # Apply simplified feature engineering
    df = engineer_features_simple(df)
    print(f"📊 After feature engineering: {len(df.columns)} columns")
    
    return df


def predict_upcoming_fights(
    upcoming_df: pd.DataFrame,
    model_path: str | Path,
    preprocessor_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Generate predictions using saved model and preprocessor."""
    
    # Load model and preprocessor
    print(f"📥 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"📥 Loading preprocessor from {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    try:
        # Use the fitted preprocessor to transform upcoming data
        X_processed = preprocessor.transform(upcoming_df)
        
        print(f"📊 Model expects {model.n_features_in_} features")
        print(f"📊 Upcoming data has {X_processed.shape[1]} features")
        
        # Check for feature alignment
        if X_processed.shape[1] != model.n_features_in_:
            raise ValueError(f"Feature mismatch: model expects {model.n_features_in_} features, got {X_processed.shape[1]}")
        
        # Make predictions
        y_proba = model.predict_proba(X_processed)[:, 1]
        y_pred = model.predict(X_processed)
        
        print(f"✅ Successfully generated predictions!")
        print(f"📊 Prediction range: {np.min(y_proba):.3f} - {np.max(y_proba):.3f}")
        print(f"📊 Mean prediction: {np.mean(y_proba):.3f}")
        print(f"📊 Std deviation: {np.std(y_proba):.3f}")
        
        # Check if predictions are diverse (not all around 50%)
        near_50_count = np.sum(np.abs(y_proba - 0.5) < 0.1)
        print(f"📊 Predictions within 10% of 50%: {near_50_count}/{len(y_proba)}")
        
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
        lambda p: max(p, 1-p)
    )
    
    # Sort by confidence
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
    
    print("🔮 Generating predictions with final model...")
    predictions = predict_upcoming_fights(
        upcoming_df=upcoming_df,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_path=args.output,
    )
    
    print("\n🏆 FINAL PREDICTIONS:")
    print("=" * 80)
    for _, row in predictions.iterrows():
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