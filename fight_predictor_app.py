"""
UFC Fight Predictor - Interactive Web Interface

A Streamlit web application where users can enter two fighters and get predictions.

Run with: streamlit run fight_predictor_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .fighter-input {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #FF6B35;
        margin: 1rem 0;
    }
    .vs-text {
        font-size: 2rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .winner-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .confidence-text {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .model-comparison {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_fighter_data():
    """Load historical fighter data for autocomplete."""
    try:
        df = pd.read_csv("data/ufc-master.csv")
        red_fighters = df['RedFighter'].dropna().unique()
        blue_fighters = df['BlueFighter'].dropna().unique()
        all_fighters = sorted(list(set(list(red_fighters) + list(blue_fighters))))
        return all_fighters, df
    except FileNotFoundError:
        st.error("Historical data not found. Please ensure data/ufc-master.csv exists.")
        return [], None

@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    try:
        if Path("models/stage1_logreg_pipeline.pkl").exists():
            models['logistic'] = joblib.load("models/stage1_logreg_pipeline.pkl")
        if Path("models/best.pkl").exists():
            models['gradient_boost'] = joblib.load("models/best.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

def get_fighter_stats(fighter_name, historical_data):
    """Get average stats for a fighter from historical data."""
    if historical_data is None:
        return None
    
    # Get all fights for this fighter (both red and blue corner)
    red_fights = historical_data[historical_data['RedFighter'] == fighter_name]
    blue_fights = historical_data[historical_data['BlueFighter'] == fighter_name]
    
    if len(red_fights) == 0 and len(blue_fights) == 0:
        return None
    
    # Calculate average stats
    stats = {}
    
    # From red corner fights
    if len(red_fights) > 0:
        for col in red_fights.columns:
            if col.startswith('Red') and red_fights[col].dtype in ['int64', 'float64']:
                base_name = col.replace('Red', '')
                if base_name not in stats:
                    stats[base_name] = []
                stats[base_name].extend(red_fights[col].dropna().tolist())
    
    # From blue corner fights
    if len(blue_fights) > 0:
        for col in blue_fights.columns:
            if col.startswith('Blue') and blue_fights[col].dtype in ['int64', 'float64']:
                base_name = col.replace('Blue', '')
                if base_name not in stats:
                    stats[base_name] = []
                stats[base_name].extend(blue_fights[col].dropna().tolist())
    
    # Calculate averages
    avg_stats = {}
    for stat, values in stats.items():
        if values:
            avg_stats[stat] = np.mean(values)
    
    return avg_stats

def create_fight_dataframe(red_fighter, blue_fighter, historical_data):
    """Create a dataframe for the hypothetical fight."""
    
    # Get fighter stats
    red_stats = get_fighter_stats(red_fighter, historical_data)
    blue_stats = get_fighter_stats(blue_fighter, historical_data)
    
    if red_stats is None or blue_stats is None:
        return None
    
    # Create fight row with all necessary columns
    fight_data = {
        'RedFighter': red_fighter,
        'BlueFighter': blue_fighter,
        'Winner': 'Red',  # Dummy value
        'Date': pd.Timestamp.now(),
        'TitleBout': False,
        'WeightClass': 'Welterweight',  # Default weight class
        'Gender': 'MALE',
        'NumberOfRounds': 3,
        'Location': 'Las Vegas, Nevada, USA',  # Default location
        'Country': 'USA',  # Default country
        'RedStance': 'Orthodox',  # Default stance
        'BlueStance': 'Orthodox',  # Default stance
        'BetterRank': 'neither',  # Default ranking
        'EmptyArena': False,
        'TotalFightTimeSecs': 900,  # 15 minutes default
        'Finish': 'No',
        'FinishDetails': '',
        'FinishRound': 3,
        'FinishRoundTime': '5:00',
    }
    
    # Add red fighter stats
    for stat, value in red_stats.items():
        fight_data[f'Red{stat}'] = value
    
    # Add blue fighter stats
    for stat, value in blue_stats.items():
        fight_data[f'Blue{stat}'] = value
    
    # Fill missing values with comprehensive defaults
    default_values = {
        # Basic fighter info
        'RedAge': 30, 'BlueAge': 30,
        'RedHeightCms': 175, 'BlueHeightCms': 175,
        'RedReachCms': 180, 'BlueReachCms': 180,
        'RedWeightLbs': 170, 'BlueWeightLbs': 170,
        
        # Fight record
        'RedWins': 10, 'BlueWins': 10,
        'RedLosses': 3, 'BlueLosses': 3,
        'RedDraws': 0, 'BlueDraws': 0,
        'RedCurrentWinStreak': 2, 'BlueCurrentWinStreak': 2,
        'RedCurrentLoseStreak': 0, 'BlueCurrentLoseStreak': 0,
        'RedLongestWinStreak': 3, 'BlueLongestWinStreak': 3,
        
        # Performance stats
        'RedAvgSigStrLanded': 4.0, 'BlueAvgSigStrLanded': 4.0,
        'RedAvgSigStrPct': 0.45, 'BlueAvgSigStrPct': 0.45,
        'RedAvgSubAtt': 0.5, 'BlueAvgSubAtt': 0.5,
        'RedAvgTDLanded': 1.5, 'BlueAvgTDLanded': 1.5,
        'RedAvgTDPct': 0.4, 'BlueAvgTDPct': 0.4,
        
        # Title bouts and rounds
        'RedTotalTitleBouts': 0, 'BlueTotalTitleBouts': 0,
        'RedTotalRoundsFought': 30, 'BlueTotalRoundsFought': 30,
        
        # Win methods
        'RedWinsByKO': 3, 'BlueWinsByKO': 3,
        'RedWinsBySubmission': 2, 'BlueWinsBySubmission': 2,
        'RedWinsByTKODoctorStoppage': 1, 'BlueWinsByTKODoctorStoppage': 1,
        'RedWinsByDecisionUnanimous': 3, 'BlueWinsByDecisionUnanimous': 3,
        'RedWinsByDecisionMajority': 1, 'BlueWinsByDecisionMajority': 1,
        'RedWinsByDecisionSplit': 0, 'BlueWinsByDecisionSplit': 0,
        
        # Odds (dummy values)
        'RedOdds': -150, 'BlueOdds': 130,
        'RedExpectedValue': 150.0, 'BlueExpectedValue': 130.0,
        'RedDecOdds': 200, 'BlueDecOdds': 250,
        'RSubOdds': 500, 'BSubOdds': 600,
        'RKOOdds': 300, 'BKOOdds': 350,
    }
    
    for key, default_val in default_values.items():
        if key not in fight_data:
            fight_data[key] = default_val
    
    # Create differential features that the pipeline expects
    fight_data['WinStreakDif'] = fight_data['RedCurrentWinStreak'] - fight_data['BlueCurrentWinStreak']
    fight_data['LoseStreakDif'] = fight_data['RedCurrentLoseStreak'] - fight_data['BlueCurrentLoseStreak']
    fight_data['WinDif'] = fight_data['RedWins'] - fight_data['BlueWins']
    fight_data['LossDif'] = fight_data['RedLosses'] - fight_data['BlueLosses']
    fight_data['HeightDif'] = fight_data['RedHeightCms'] - fight_data['BlueHeightCms']
    fight_data['ReachDif'] = fight_data['RedReachCms'] - fight_data['BlueReachCms']
    fight_data['AgeDif'] = fight_data['RedAge'] - fight_data['BlueAge']
    fight_data['SigStrDif'] = fight_data['RedAvgSigStrLanded'] - fight_data['BlueAvgSigStrLanded']
    fight_data['AvgSubAttDif'] = fight_data['RedAvgSubAtt'] - fight_data['BlueAvgSubAtt']
    fight_data['AvgTDDif'] = fight_data['RedAvgTDLanded'] - fight_data['BlueAvgTDLanded']
    fight_data['TotalRoundDif'] = fight_data['RedTotalRoundsFought'] - fight_data['BlueTotalRoundsFought']
    fight_data['TotalTitleBoutDif'] = fight_data['RedTotalTitleBouts'] - fight_data['BlueTotalTitleBouts']
    fight_data['KODif'] = fight_data['RedWinsByKO'] - fight_data['BlueWinsByKO']
    fight_data['SubDif'] = fight_data['RedWinsBySubmission'] - fight_data['BlueWinsBySubmission']
    fight_data['LongestWinStreakDif'] = fight_data['RedLongestWinStreak'] - fight_data['BlueLongestWinStreak']
    
    return pd.DataFrame([fight_data])

def predict_fight(red_fighter, blue_fighter, models, historical_data):
    """Predict the outcome of a fight between two fighters."""
    
    # Create fight dataframe
    fight_df = create_fight_dataframe(red_fighter, blue_fighter, historical_data)
    
    if fight_df is None:
        return None
    
    predictions = {}
    
    # Try each model
    for model_name, model in models.items():
        try:
            if model_name == 'logistic':
                # Pipeline model - needs full preprocessing pipeline
                # Apply same preprocessing as training data
                from scripts.preprocess import _coerce_dates, _create_target, _drop_high_missing
                from scripts.features import engineer_features
                
                # Create a copy for processing
                fight_processed = fight_df.copy()
                
                # Apply preprocessing steps
                fight_processed = _coerce_dates(fight_processed)
                fight_processed = _create_target(fight_processed)
                fight_processed = _drop_high_missing(fight_processed, threshold=0.7)
                
                # Apply feature engineering
                fight_processed = engineer_features(fight_processed)
                
                # Drop target column for prediction
                X = fight_processed.drop(columns=['target'])
                
                # Make prediction
                prob = model.predict_proba(X)[0, 1]  # Probability of red winning
                winner = 'Red' if prob > 0.5 else 'Blue'
                confidence = max(prob, 1-prob)
                
            else:
                # Gradient boost model - needs preprocessing
                from scripts.preprocess import _coerce_dates, _create_target, _drop_high_missing, scale_features
                from scripts.features import engineer_features
                
                # Create a copy for processing
                fight_processed = fight_df.copy()
                
                # Apply preprocessing steps (same as training)
                fight_processed = _coerce_dates(fight_processed)
                fight_processed = _create_target(fight_processed)
                fight_processed = _drop_high_missing(fight_processed, threshold=0.7)
                fight_processed = fight_processed.reset_index(drop=True)
                
                # Apply feature engineering
                fight_processed = engineer_features(fight_processed)
                
                # Scale features
                X_processed, _ = scale_features(fight_processed)
                
                # Handle feature mismatch by padding
                if X_processed.shape[1] < 4415:
                    padding = np.zeros((X_processed.shape[0], 4415 - X_processed.shape[1]))
                    X_processed = np.hstack([X_processed, padding])
                
                # Make prediction
                prob = model.predict_proba(X_processed)[0, 1]
                winner = 'Red' if prob > 0.5 else 'Blue'
                confidence = max(prob, 1-prob)
            
            predictions[model_name] = {
                'winner': winner,
                'red_prob': prob,
                'confidence': confidence
            }
            
        except Exception as e:
            st.error(f"Error with {model_name} model: {e}")
            continue
    
    return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">🥊 UFC Fight Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Enter two fighters to predict who would win!</p>', unsafe_allow_html=True)
    
    # Load data and models
    fighters_list, historical_data = load_fighter_data()
    models = load_models()
    
    if not models:
        st.error("No trained models found. Please train models first.")
        st.stop()
    
    # Fighter selection
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown('<div class="fighter-input">', unsafe_allow_html=True)
        st.markdown("### 🔴 Red Corner")
        red_fighter = st.selectbox(
            "Select Red Fighter:",
            options=[""] + fighters_list,
            key="red_fighter",
            help="Choose the fighter for the red corner"
        )
        if red_fighter and red_fighter in fighters_list:
            red_stats = get_fighter_stats(red_fighter, historical_data)
            if red_stats:
                st.caption(f"Avg Wins: {red_stats.get('Wins', 'N/A')}")
                st.caption(f"Avg Sig Strikes: {red_stats.get('AvgSigStrLanded', 'N/A'):.1f}" if red_stats.get('AvgSigStrLanded') else "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="fighter-input">', unsafe_allow_html=True)
        st.markdown("### 🔵 Blue Corner")
        blue_fighter = st.selectbox(
            "Select Blue Fighter:",
            options=[""] + fighters_list,
            key="blue_fighter",
            help="Choose the fighter for the blue corner"
        )
        if blue_fighter and blue_fighter in fighters_list:
            blue_stats = get_fighter_stats(blue_fighter, historical_data)
            if blue_stats:
                st.caption(f"Avg Wins: {blue_stats.get('Wins', 'N/A')}")
                st.caption(f"Avg Sig Strikes: {blue_stats.get('AvgSigStrLanded', 'N/A'):.1f}" if blue_stats.get('AvgSigStrLanded') else "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    if st.button("🥊 PREDICT FIGHT OUTCOME", type="primary", use_container_width=True):
        if not red_fighter or not blue_fighter:
            st.warning("Please select both fighters!")
        elif red_fighter == blue_fighter:
            st.warning("Please select different fighters!")
        else:
            with st.spinner("Analyzing fighters and predicting outcome..."):
                predictions = predict_fight(red_fighter, blue_fighter, models, historical_data)
                
                if predictions:
                    # Show main prediction (use gradient boost if available, otherwise logistic)
                    main_model = 'gradient_boost' if 'gradient_boost' in predictions else 'logistic'
                    main_pred = predictions[main_model]
                    
                    # Main prediction card
                    winner_name = red_fighter if main_pred['winner'] == 'Red' else blue_fighter
                    winner_color = "🔴" if main_pred['winner'] == 'Red' else "🔵"
                    
                    st.markdown(f'''
                    <div class="prediction-card">
                        <div class="winner-text">{winner_color} {winner_name} WINS!</div>
                        <div class="confidence-text">Confidence: {main_pred['confidence']:.1%}</div>
                        <div style="margin-top: 1rem; font-size: 1.1rem;">
                            Red Win Probability: {main_pred['red_prob']:.1%}<br>
                            Blue Win Probability: {1-main_pred['red_prob']:.1%}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Model comparison if multiple models
                    if len(predictions) > 1:
                        st.markdown("### 📊 Model Comparison")
                        
                        for model_name, pred in predictions.items():
                            model_display = "Logistic Regression" if model_name == 'logistic' else "Gradient Boosting"
                            winner_name = red_fighter if pred['winner'] == 'Red' else blue_fighter
                            winner_emoji = "🔴" if pred['winner'] == 'Red' else "🔵"
                            
                            st.markdown(f'''
                            <div class="model-comparison">
                                <strong>{model_display}</strong><br>
                                Winner: {winner_emoji} {winner_name} ({pred['confidence']:.1%} confidence)<br>
                                Red: {pred['red_prob']:.1%} | Blue: {1-pred['red_prob']:.1%}
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # Fight breakdown
                    st.markdown("### 🔍 Fight Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{red_fighter}** (Red Corner)")
                        red_stats = get_fighter_stats(red_fighter, historical_data)
                        if red_stats:
                            st.write(f"• Avg Wins: {red_stats.get('Wins', 'N/A')}")
                            st.write(f"• Avg Sig Strikes: {red_stats.get('AvgSigStrLanded', 'N/A'):.1f}" if red_stats.get('AvgSigStrLanded') else "• Sig Strikes: N/A")
                            st.write(f"• Avg Takedowns: {red_stats.get('AvgTDLanded', 'N/A'):.1f}" if red_stats.get('AvgTDLanded') else "• Takedowns: N/A")
                    
                    with col2:
                        st.markdown(f"**{blue_fighter}** (Blue Corner)")
                        blue_stats = get_fighter_stats(blue_fighter, historical_data)
                        if blue_stats:
                            st.write(f"• Avg Wins: {blue_stats.get('Wins', 'N/A')}")
                            st.write(f"• Avg Sig Strikes: {blue_stats.get('AvgSigStrLanded', 'N/A'):.1f}" if blue_stats.get('AvgSigStrLanded') else "• Sig Strikes: N/A")
                            st.write(f"• Avg Takedowns: {blue_stats.get('AvgTDLanded', 'N/A'):.1f}" if blue_stats.get('AvgTDLanded') else "• Takedowns: N/A")
                
                else:
                    st.error("Could not generate predictions. One or both fighters may not have enough historical data.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by Machine Learning | 📊 Based on Historical UFC Data</p>
        <p><small>Predictions are for entertainment purposes only</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()