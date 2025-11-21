"""
UFC Fight Predictor - Interactive Web Interface (Redesigned)

A modern, beautiful Streamlit web application where users can enter two fighters and get predictions.

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
    page_title="UFC Fight Predictor | Interactive",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF0000 0%, #000000 50%, #0066FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Fighter Input Cards */
    .fighter-input-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 2px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .fighter-input-card.red {
        border-color: #DC2626;
    }
    
    .fighter-input-card.blue {
        border-color: #2563EB;
    }
    
    .fighter-input-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .fighter-input-card.red::before {
        background: linear-gradient(90deg, #DC2626 0%, #991B1B 100%);
    }
    
    .fighter-input-card.blue::before {
        background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
    }
    
    .fighter-input-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.08);
    }
    
    .corner-label {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .corner-label.red {
        color: #DC2626;
    }
    
    .corner-label.blue {
        color: #2563EB;
    }
    
    .vs-text {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: white;
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #FF0000 0%, #000000 50%, #0066FF 100%);
    }
    
    .winner-text {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #e5e7eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        color: #d1d5db;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    .probability-breakdown {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .probability-item {
        text-align: center;
    }
    
    .probability-label {
        font-size: 0.875rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .probability-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    
    /* Model Comparison */
    .model-comparison-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .model-comparison-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    
    .model-name {
        font-size: 1.125rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    .model-prediction {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .prediction-badge.winner {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
    }
    
    .prediction-badge.loser {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    /* Stats Display */
    .stats-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .fighter-name-stats {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .fighter-name-stats.red {
        color: #DC2626;
    }
    
    .fighter-name-stats.blue {
        color: #2563EB;
    }
    
    .stat-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
    }
    
    .stat-item:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        color: #6b7280;
        font-weight: 500;
    }
    
    .stat-value {
        color: #111827;
        font-weight: 700;
    }
    
    /* Confidence Bar */
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: #e5e7eb;
        overflow: hidden;
        margin-top: 0.75rem;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill.high {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
    }
    
    .confidence-fill.medium {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
    }
    
    .confidence-fill.low {
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        if Path("models/gbdt_final.pkl").exists():
            models['gradient_boost'] = joblib.load("models/gbdt_final.pkl")
        elif Path("models/best.pkl").exists():
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
                # Gradient boost model - use simplified preprocessing
                try:
                    # Load the preprocessor for the new model
                    preprocessor_path = "models/gbdt_final_preprocessor.pkl"
                    if Path(preprocessor_path).exists():
                        preprocessor = joblib.load(preprocessor_path)
                        
                        # Apply simplified preprocessing
                        from scripts.preprocess_simple import _coerce_dates, _create_target, _drop_high_missing
                        from scripts.features_simple import engineer_features_simple
                        
                        fight_processed = fight_df.copy()
                        fight_processed = _coerce_dates(fight_processed)
                        fight_processed = _create_target(fight_processed)
                        
                        # Preserve important columns before dropping high missing
                        important_cols = ['Finish', 'FinishRound', 'TotalFightTimeSecs', 'EmptyArena']
                        preserved_data = {}
                        for col in important_cols:
                            if col in fight_processed.columns:
                                preserved_data[col] = fight_processed[col].copy()
                        
                        fight_processed = _drop_high_missing(fight_processed, threshold=0.7)
                        
                        # Restore important columns
                        for col, data in preserved_data.items():
                            if col not in fight_processed.columns:
                                fight_processed[col] = data
                        
                        fight_processed = fight_processed.reset_index(drop=True)
                        fight_processed = engineer_features_simple(fight_processed)
                        
                        # Use the fitted preprocessor
                        X_processed = preprocessor.transform(fight_processed)
                        
                        # Make prediction
                        prob = model.predict_proba(X_processed)[0, 1]
                        winner = 'Red' if prob > 0.5 else 'Blue'
                        confidence = max(prob, 1-prob)
                        
                    else:
                        # Fallback to old method if preprocessor not found
                        st.error("New preprocessor not found. Please retrain the model.")
                        continue
                        
                except Exception as e:
                    st.error(f"Error with simplified preprocessing: {e}")
                    continue
            
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
    st.markdown('<p class="sub-header">Enter two fighters to predict who would win!</p>', unsafe_allow_html=True)
    
    # Load data and models
    fighters_list, historical_data = load_fighter_data()
    models = load_models()
    
    if not models:
        st.error("No trained models found. Please train models first.")
        st.stop()
    
    # Fighter selection
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown('<div class="fighter-input-card red">', unsafe_allow_html=True)
        st.markdown('<div class="corner-label red">🔴 Red Corner</div>', unsafe_allow_html=True)
        red_fighter = st.selectbox(
            "Select Red Fighter:",
            options=[""] + fighters_list,
            key="red_fighter",
            help="Choose the fighter for the red corner",
            label_visibility="collapsed"
        )
        if red_fighter and red_fighter in fighters_list:
            red_stats = get_fighter_stats(red_fighter, historical_data)
            if red_stats:
                st.caption(f"📊 Avg Wins: {red_stats.get('Wins', 'N/A'):.0f}" if red_stats.get('Wins') else "📊 Avg Wins: N/A")
                st.caption(f"👊 Avg Sig Strikes: {red_stats.get('AvgSigStrLanded', 'N/A'):.1f}" if red_stats.get('AvgSigStrLanded') else "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="fighter-input-card blue">', unsafe_allow_html=True)
        st.markdown('<div class="corner-label blue">🔵 Blue Corner</div>', unsafe_allow_html=True)
        blue_fighter = st.selectbox(
            "Select Blue Fighter:",
            options=[""] + fighters_list,
            key="blue_fighter",
            help="Choose the fighter for the blue corner",
            label_visibility="collapsed"
        )
        if blue_fighter and blue_fighter in fighters_list:
            blue_stats = get_fighter_stats(blue_fighter, historical_data)
            if blue_stats:
                st.caption(f"📊 Avg Wins: {blue_stats.get('Wins', 'N/A'):.0f}" if blue_stats.get('Wins') else "📊 Avg Wins: N/A")
                st.caption(f"👊 Avg Sig Strikes: {blue_stats.get('AvgSigStrLanded', 'N/A'):.1f}" if blue_stats.get('AvgSigStrLanded') else "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🥊 PREDICT FIGHT OUTCOME", type="primary", use_container_width=True):
        if not red_fighter or not blue_fighter:
            st.warning("⚠️ Please select both fighters!")
        elif red_fighter == blue_fighter:
            st.warning("⚠️ Please select different fighters!")
        else:
            with st.spinner("🔮 Analyzing fighters and predicting outcome..."):
                predictions = predict_fight(red_fighter, blue_fighter, models, historical_data)
                
                if predictions:
                    # Show main prediction (use gradient boost if available, otherwise logistic)
                    main_model = 'gradient_boost' if 'gradient_boost' in predictions else 'logistic'
                    main_pred = predictions[main_model]
                    
                    # Main prediction card
                    winner_name = red_fighter if main_pred['winner'] == 'Red' else blue_fighter
                    winner_color = "🔴" if main_pred['winner'] == 'Red' else "🔵"
                    conf_level = "high" if main_pred['confidence'] > 0.7 else "medium" if main_pred['confidence'] > 0.55 else "low"
                    
                    st.markdown(f'''
                    <div class="prediction-card">
                        <div class="winner-text">{winner_color} {winner_name} WINS!</div>
                        <div class="confidence-text">Confidence: {main_pred['confidence']:.1%}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill {conf_level}" style="width: {main_pred['confidence']*100}%"></div>
                        </div>
                        <div class="probability-breakdown">
                            <div class="probability-item">
                                <div class="probability-label">🔴 Red Corner</div>
                                <div class="probability-value">{main_pred['red_prob']:.1%}</div>
                            </div>
                            <div class="probability-item">
                                <div class="probability-label">🔵 Blue Corner</div>
                                <div class="probability-value">{1-main_pred['red_prob']:.1%}</div>
                            </div>
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
                            conf_level = "high" if pred['confidence'] > 0.7 else "medium" if pred['confidence'] > 0.55 else "low"
                            
                            st.markdown(f'''
                            <div class="model-comparison-card">
                                <div class="model-name">{model_display}</div>
                                <div class="model-prediction">
                                    <span class="prediction-badge winner">{winner_emoji} {winner_name}</span>
                                    <span style="color: #6b7280; font-weight: 600;">{pred['confidence']:.1%} confidence</span>
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                                    <div>
                                        <div style="color: #6b7280; font-size: 0.875rem;">Red Win Probability</div>
                                        <div style="color: #DC2626; font-weight: 700; font-size: 1.25rem;">{pred['red_prob']:.1%}</div>
                                    </div>
                                    <div>
                                        <div style="color: #6b7280; font-size: 0.875rem;">Blue Win Probability</div>
                                        <div style="color: #2563EB; font-weight: 700; font-size: 1.25rem;">{1-pred['red_prob']:.1%}</div>
                                    </div>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill {conf_level}" style="width: {pred['confidence']*100}%"></div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # Fight breakdown
                    st.markdown("### 🔍 Fighter Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        red_stats = get_fighter_stats(red_fighter, historical_data)
                        if red_stats:
                            stats_html = f'''
                            <div class="stats-card">
                                <div class="fighter-name-stats red">🔴 {red_fighter}</div>
                                <div class="stat-item">
                                    <span class="stat-label">Average Wins</span>
                                    <span class="stat-value">{red_stats.get('Wins', 'N/A'):.0f if red_stats.get('Wins') else 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Avg Sig Strikes</span>
                                    <span class="stat-value">{red_stats.get('AvgSigStrLanded', 'N/A'):.1f if red_stats.get('AvgSigStrLanded') else 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Avg Takedowns</span>
                                    <span class="stat-value">{red_stats.get('AvgTDLanded', 'N/A'):.1f if red_stats.get('AvgTDLanded') else 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Win Streak</span>
                                    <span class="stat-value">{red_stats.get('CurrentWinStreak', 'N/A'):.0f if red_stats.get('CurrentWinStreak') else 'N/A'}</span>
                                </div>
                            </div>
                            '''
                            st.markdown(stats_html, unsafe_allow_html=True)
                    
                    with col2:
                        blue_stats = get_fighter_stats(blue_fighter, historical_data)
                        if blue_stats:
                            stats_html = f'''
                            <div class="stats-card">
                                <div class="fighter-name-stats blue">🔵 {blue_fighter}</div>
                                <div class="stat-item">
                                    <span class="stat-label">Average Wins</span>
                                    <span class="stat-value">{blue_stats.get('Wins', 'N/A'):.0f if blue_stats.get('Wins') else 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Avg Sig Strikes</span>
                                    <span class="stat-value">{blue_stats.get('AvgSigStrLanded', 'N/A'):.1f if blue_stats.get('AvgSigStrLanded') else 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Avg Takedowns</span>
                                    <span class="stat-value">{blue_stats.get('AvgTDLanded', 'N/A'):.1f if blue_stats.get('AvgTDLanded') else 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Win Streak</span>
                                    <span class="stat-value">{blue_stats.get('CurrentWinStreak', 'N/A'):.0f if blue_stats.get('CurrentWinStreak') else 'N/A'}</span>
                                </div>
                            </div>
                            '''
                            st.markdown(stats_html, unsafe_allow_html=True)
                
                else:
                    st.error("❌ Could not generate predictions. One or both fighters may not have enough historical data.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; margin: 2rem 0; padding: 1rem;">
        <p style="font-size: 0.875rem; margin: 0.5rem 0;">
            🤖 Powered by Machine Learning | 📊 Based on Historical UFC Data
        </p>
        <p style="font-size: 0.75rem; margin: 0.5rem 0; color: #9ca3af;">
            Predictions are for entertainment purposes only
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
