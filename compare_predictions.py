"""
compare_predictions.py

Compare predictions from both the logistic regression pipeline and gradient boosting models.
"""

import pandas as pd

def main():
    # Load both prediction files
    logreg_preds = pd.read_csv("predictions.csv")
    gbdt_preds = pd.read_csv("predictions_gbdt.csv")
    
    # Merge on fighter names
    comparison = pd.merge(
        logreg_preds, 
        gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    print("🥊 UFC Fight Predictions Comparison")
    print("=" * 80)
    print(f"{'Fight':<35} {'LogReg':<15} {'GradBoost':<15} {'Agreement':<10}")
    print("-" * 80)
    
    agreements = 0
    total_fights = len(comparison)
    
    for _, row in comparison.iterrows():
        fight = f"{row['RedFighter']} vs {row['BlueFighter']}"
        if len(fight) > 34:
            fight = fight[:31] + "..."
        
        logreg_pred = f"{row['predicted_winner_logreg']} ({row['prob_red_win_logreg']:.3f})"
        gbdt_pred = f"{row['predicted_winner_gbdt']} ({row['prob_red_win_gbdt']:.3f})"
        
        agree = "✅" if row['predicted_winner_logreg'] == row['predicted_winner_gbdt'] else "❌"
        if row['predicted_winner_logreg'] == row['predicted_winner_gbdt']:
            agreements += 1
        
        print(f"{fight:<35} {logreg_pred:<15} {gbdt_pred:<15} {agree:<10}")
    
    print("-" * 80)
    print(f"Model Agreement: {agreements}/{total_fights} ({agreements/total_fights*100:.1f}%)")
    
    # Show biggest disagreements
    comparison['prob_diff'] = abs(comparison['prob_red_win_logreg'] - comparison['prob_red_win_gbdt'])
    biggest_disagreements = comparison.nlargest(3, 'prob_diff')
    
    print("\n🤔 Biggest Disagreements:")
    print("=" * 50)
    for _, row in biggest_disagreements.iterrows():
        fight = f"{row['RedFighter']} vs {row['BlueFighter']}"
        print(f"\n{fight}")
        print(f"  LogReg:     {row['predicted_winner_logreg']} (Red: {row['prob_red_win_logreg']:.3f})")
        print(f"  GradBoost:  {row['predicted_winner_gbdt']} (Red: {row['prob_red_win_gbdt']:.3f})")
        print(f"  Difference: {row['prob_diff']:.3f}")

if __name__ == "__main__":
    main()