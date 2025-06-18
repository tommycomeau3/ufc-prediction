#!/usr/bin/env python3
"""
Trace exactly where the parquet error occurs in matchup.py
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

print("=== Tracing matchup.py execution ===")

try:
    print("1. Importing modules...")
    import pandas as pd
    print("   ✅ pandas imported successfully")
    
    import joblib
    print("   ✅ joblib imported successfully")
    
    import numpy as np
    print("   ✅ numpy imported successfully")
    
    print("2. Importing scripts.features...")
    from scripts.features import engineer_features
    print("   ✅ scripts.features imported successfully")
    
    print("3. Importing scripts.matchup functions...")
    from scripts.matchup import load_stats_cache, predict_matchup
    print("   ✅ scripts.matchup imported successfully")
    
    print("4. Testing load_stats_cache()...")
    stats_df = load_stats_cache()
    print(f"   ✅ Stats cache loaded successfully: {stats_df.shape}")
    
    print("5. Testing predict_matchup()...")
    p1, p2 = predict_matchup("Conor McGregor", "Khabib Nurmagomedov")
    print(f"   ✅ Prediction successful: {p1:.2%} vs {p2:.2%}")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\n=== Full traceback ===")
    traceback.print_exc()
    
    print("\n=== Analyzing the error ===")
    error_str = str(e)
    if "parquet" in error_str.lower():
        print("🔍 This is definitely a parquet-related error")
        print("🔍 The error is occurring somewhere in the data processing pipeline")
    else:
        print("🔍 This might not be a parquet error after all")