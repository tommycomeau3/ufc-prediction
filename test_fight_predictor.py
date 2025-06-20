"""
Test script for the fight predictor functionality
"""

import pandas as pd
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

def test_fight_prediction():
    """Test the fight prediction functionality."""
    
    # Import the functions from the app
    from fight_predictor_app import load_fighter_data, load_models, predict_fight
    
    print("🧪 Testing UFC Fight Predictor...")
    
    # Load data and models
    print("📥 Loading fighter data and models...")
    fighters_list, historical_data = load_fighter_data()
    models = load_models()
    
    if not fighters_list:
        print("❌ No fighter data found")
        return False
    
    if not models:
        print("❌ No models found")
        return False
    
    print(f"✅ Loaded {len(fighters_list)} fighters")
    print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
    
    # Test with some fighters
    test_fighters = [
        ("Jon Jones", "Daniel Cormier"),
        ("Conor McGregor", "Nate Diaz"),
        ("Amanda Nunes", "Ronda Rousey")
    ]
    
    for red_fighter, blue_fighter in test_fighters:
        if red_fighter in fighters_list and blue_fighter in fighters_list:
            print(f"\n🥊 Testing: {red_fighter} vs {blue_fighter}")
            
            try:
                predictions = predict_fight(red_fighter, blue_fighter, models, historical_data)
                
                if predictions:
                    for model_name, pred in predictions.items():
                        model_display = "Logistic Regression" if model_name == 'logistic' else "Gradient Boosting"
                        winner = red_fighter if pred['winner'] == 'Red' else blue_fighter
                        print(f"  {model_display}: {winner} wins ({pred['confidence']:.1%} confidence)")
                    print("  ✅ Prediction successful")
                else:
                    print("  ❌ No predictions generated")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            break  # Test just one fight
    
    print("\n🎉 Test completed!")
    return True

if __name__ == "__main__":
    test_fight_prediction()