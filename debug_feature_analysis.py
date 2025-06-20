#!/usr/bin/env python3
"""
Debug script to analyze feature mismatch and model behavior.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from scripts.preprocess import clean_ufc_data, scale_features
from scripts.features import engineer_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch between training and prediction data."""
    
    print("🔍 FEATURE MISMATCH ANALYSIS")
    print("=" * 50)
    
    # Load the gradient boosting model
    model_path = "models/best.pkl"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model from {model_path}")
    
    # Check expected features
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
        print(f"📊 Model expects {expected_features} features")
    else:
        print("⚠️  Cannot determine expected feature count from model")
        return
    
    # Load and process training data to see actual feature count
    print("\n🏋️  Processing training data...")
    df = clean_ufc_data("data/ufc-master.csv")
    df = engineer_features(df)
    X_train, y_train = scale_features(df)
    
    actual_features = X_train.shape[1]
    print(f"📊 Training data has {actual_features} features")
    print(f"📊 Feature mismatch: {expected_features - actual_features} features")
    
    # Load and process upcoming data
    print("\n🔮 Processing upcoming data...")
    upcoming_df = pd.read_csv("data/upcoming.csv")
    
    # Add dummy Winner column
    upcoming_df["Winner"] = "Red"
    
    # Apply same preprocessing
    from scripts.preprocess import _coerce_dates, _create_target, _drop_high_missing
    upcoming_df = _coerce_dates(upcoming_df)
    upcoming_df = _create_target(upcoming_df)
    upcoming_df = _drop_high_missing(upcoming_df, threshold=0.7)
    upcoming_df = upcoming_df.reset_index(drop=True)
    
    # Apply feature engineering
    upcoming_df = engineer_features(upcoming_df)
    
    # Scale features
    X_upcoming, _ = scale_features(upcoming_df)
    upcoming_features = X_upcoming.shape[1]
    
    print(f"📊 Upcoming data has {upcoming_features} features")
    print(f"📊 Missing features: {expected_features - upcoming_features}")
    
    # Test predictions with and without padding
    print("\n🧪 TESTING PREDICTIONS")
    print("=" * 30)
    
    # Test 1: With zero padding (current approach)
    if upcoming_features < expected_features:
        padding = np.zeros((X_upcoming.shape[0], expected_features - upcoming_features))
        X_padded = np.hstack([X_upcoming, padding])
        
        probs_padded = model.predict_proba(X_padded)[:, 1]
        print(f"🔴 With zero padding:")
        print(f"   Mean probability: {np.mean(probs_padded):.3f}")
        print(f"   Std deviation: {np.std(probs_padded):.3f}")
        print(f"   Min: {np.min(probs_padded):.3f}, Max: {np.max(probs_padded):.3f}")
        
        # Count predictions near 50%
        near_50_count = np.sum(np.abs(probs_padded - 0.5) < 0.1)
        print(f"   Predictions within 10% of 50%: {near_50_count}/{len(probs_padded)}")
    
    # Test 2: Using only available features (truncated model)
    print(f"\n🔵 Feature analysis:")
    print(f"   Zero-padded features: {expected_features - upcoming_features}")
    print(f"   Percentage of features that are zeros: {(expected_features - upcoming_features) / expected_features * 100:.1f}%")
    
    return {
        'expected_features': expected_features,
        'actual_features': upcoming_features,
        'zero_padded_features': expected_features - upcoming_features,
        'predictions': probs_padded if 'probs_padded' in locals() else None
    }

def test_model_sensitivity():
    """Test how sensitive the model is to zero-padding."""
    
    print("\n🧪 MODEL SENSITIVITY TEST")
    print("=" * 30)
    
    model = joblib.load("models/best.pkl")
    
    # Create a simple test case with known features
    np.random.seed(42)
    n_real_features = 143
    n_total_features = 4415
    n_samples = 5
    
    # Create realistic feature values
    X_real = np.random.normal(0, 1, (n_samples, n_real_features))
    
    # Test different padding strategies
    strategies = {
        'zero_padding': np.zeros((n_samples, n_total_features - n_real_features)),
        'mean_padding': np.full((n_samples, n_total_features - n_real_features), 0.0),
        'small_noise': np.random.normal(0, 0.01, (n_samples, n_total_features - n_real_features))
    }
    
    for strategy_name, padding in strategies.items():
        X_test = np.hstack([X_real, padding])
        probs = model.predict_proba(X_test)[:, 1]
        
        print(f"📊 {strategy_name}:")
        print(f"   Mean: {np.mean(probs):.3f}, Std: {np.std(probs):.3f}")
        print(f"   Range: {np.min(probs):.3f} - {np.max(probs):.3f}")

if __name__ == "__main__":
    results = analyze_feature_mismatch()
    test_model_sensitivity()
    
    print("\n🎯 CONCLUSION")
    print("=" * 20)
    if results and results['zero_padded_features'] > results['actual_features']:
        print("❌ CRITICAL ISSUE FOUND:")
        print(f"   You're padding {results['zero_padded_features']} features with zeros!")
        print(f"   This is {results['zero_padded_features'] / results['expected_features'] * 100:.1f}% of all features!")
        print("   This explains why predictions are near 50% - the model can't learn from mostly-zero data.")
    else:
        print("✅ No obvious feature mismatch issues found.")