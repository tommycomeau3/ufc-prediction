"""
UFC Fight Prediction Dashboard

A Streamlit web application for visualizing UFC fight predictions and model performance.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import joblib

# Page configuration
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .fight-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictions():
    """Load prediction data from CSV files."""
    try:
        logreg_preds = pd.read_csv("predictions.csv")
        gbdt_preds = pd.read_csv("predictions_gbdt.csv")
        return logreg_preds, gbdt_preds
    except FileNotFoundError:
        st.error("Prediction files not found. Please run the prediction scripts first.")
        return None, None

@st.cache_data
def load_upcoming_data():
    """Load upcoming fights data."""
    try:
        return pd.read_csv("data/upcoming.csv")
    except FileNotFoundError:
        st.error("Upcoming fights data not found.")
        return None

def create_confidence_chart(logreg_preds, gbdt_preds):
    """Create a confidence comparison chart."""
    # Merge predictions
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    # Create fight labels
    merged['fight'] = merged['RedFighter'] + ' vs ' + merged['BlueFighter']
    
    fig = go.Figure()
    
    # Add LogReg confidence
    fig.add_trace(go.Bar(
        name='Logistic Regression',
        x=merged['fight'],
        y=merged['confidence_logreg'],
        marker_color='#1f77b4',
        text=[f"{c:.1%}" for c in merged['confidence_logreg']],
        textposition='auto',
    ))
    
    # Add GradBoost confidence
    fig.add_trace(go.Bar(
        name='Gradient Boosting',
        x=merged['fight'],
        y=merged['confidence_gbdt'],
        marker_color='#ff7f0e',
        text=[f"{c:.1%}" for c in merged['confidence_gbdt']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Model Confidence Comparison",
        xaxis_title="Fights",
        yaxis_title="Confidence",
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def create_agreement_chart(logreg_preds, gbdt_preds):
    """Create a model agreement visualization."""
    # Merge predictions
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    # Calculate agreement
    merged['agrees'] = merged['predicted_winner_logreg'] == merged['predicted_winner_gbdt']
    agreement_rate = merged['agrees'].mean()
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Agree', 'Disagree'],
        values=[merged['agrees'].sum(), (~merged['agrees']).sum()],
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    
    fig.update_layout(
        title=f"Model Agreement Rate: {agreement_rate:.1%}",
        height=400
    )
    
    return fig

def create_probability_scatter(logreg_preds, gbdt_preds):
    """Create scatter plot comparing probabilities."""
    # Merge predictions
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    merged['fight'] = merged['RedFighter'] + ' vs ' + merged['BlueFighter']
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=merged['prob_red_win_logreg'],
        y=merged['prob_red_win_gbdt'],
        mode='markers+text',
        text=merged['fight'],
        textposition="top center",
        marker=dict(
            size=10,
            color=merged['confidence_logreg'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="LogReg Confidence")
        ),
        name='Fights'
    ))
    
    # Add diagonal line (perfect agreement)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Perfect Agreement'
    ))
    
    fig.update_layout(
        title="Red Win Probability Comparison",
        xaxis_title="Logistic Regression Probability",
        yaxis_title="Gradient Boosting Probability",
        height=500
    )
    
    return fig

def display_fight_cards(logreg_preds, gbdt_preds, upcoming_data):
    """Display individual fight prediction cards."""
    # Merge all data
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    if upcoming_data is not None:
        merged = pd.merge(
            merged, upcoming_data[['RedFighter', 'BlueFighter', 'WeightClass', 'Date']],
            on=["RedFighter", "BlueFighter"],
            how='left'
        )
    
    # Sort by highest confidence
    merged = merged.sort_values('confidence_gbdt', ascending=False)
    
    for _, fight in merged.iterrows():
        with st.container():
            st.markdown('<div class="fight-card">', unsafe_allow_html=True)
            
            # Fight header
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"### 🔴 {fight['RedFighter']}")
                
            with col2:
                st.markdown("### VS")
                if 'WeightClass' in fight:
                    st.caption(f"{fight['WeightClass']}")
                if 'Date' in fight:
                    st.caption(f"{fight['Date']}")
                    
            with col3:
                st.markdown(f"### 🔵 {fight['BlueFighter']}")
            
            # Predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Logistic Regression**")
                winner_lr = fight['predicted_winner_logreg']
                prob_lr = fight['prob_red_win_logreg']
                conf_lr = fight['confidence_logreg']
                
                if winner_lr == 'Red':
                    st.success(f"🔴 {fight['RedFighter']} ({prob_lr:.1%})")
                else:
                    st.info(f"🔵 {fight['BlueFighter']} ({1-prob_lr:.1%})")
                st.caption(f"Confidence: {conf_lr:.1%}")
                
            with col2:
                st.markdown("**Gradient Boosting**")
                winner_gb = fight['predicted_winner_gbdt']
                prob_gb = fight['prob_red_win_gbdt']
                conf_gb = fight['confidence_gbdt']
                
                if winner_gb == 'Red':
                    st.success(f"🔴 {fight['RedFighter']} ({prob_gb:.1%})")
                else:
                    st.info(f"🔵 {fight['BlueFighter']} ({1-prob_gb:.1%})")
                st.caption(f"Confidence: {conf_gb:.1%}")
            
            # Agreement indicator
            if winner_lr == winner_gb:
                st.success("✅ Models Agree")
            else:
                st.warning("❌ Models Disagree")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">🥊 UFC Fight Predictor Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    logreg_preds, gbdt_preds = load_predictions()
    upcoming_data = load_upcoming_data()
    
    if logreg_preds is None or gbdt_preds is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Model Comparison", "Fight Predictions", "Model Performance"]
    )
    
    if page == "Overview":
        st.header("📊 Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Fights", len(logreg_preds))
            
        with col2:
            # Calculate agreement
            merged = pd.merge(logreg_preds, gbdt_preds, on=["RedFighter", "BlueFighter"], suffixes=("_lr", "_gb"))
            agreement = (merged['predicted_winner_lr'] == merged['predicted_winner_gb']).mean()
            st.metric("Model Agreement", f"{agreement:.1%}")
            
        with col3:
            avg_conf_lr = logreg_preds['confidence'].mean()
            st.metric("Avg LogReg Confidence", f"{avg_conf_lr:.1%}")
            
        with col4:
            avg_conf_gb = gbdt_preds['confidence'].mean()
            st.metric("Avg GradBoost Confidence", f"{avg_conf_gb:.1%}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_agreement = create_agreement_chart(logreg_preds, gbdt_preds)
            st.plotly_chart(fig_agreement, use_container_width=True)
            
        with col2:
            fig_confidence = create_confidence_chart(logreg_preds, gbdt_preds)
            st.plotly_chart(fig_confidence, use_container_width=True)
    
    elif page == "Model Comparison":
        st.header("🔍 Model Comparison")
        
        # Probability scatter plot
        fig_scatter = create_probability_scatter(logreg_preds, gbdt_preds)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Disagreement analysis
        st.subheader("Biggest Disagreements")
        merged = pd.merge(logreg_preds, gbdt_preds, on=["RedFighter", "BlueFighter"], suffixes=("_lr", "_gb"))
        merged['prob_diff'] = abs(merged['prob_red_win_lr'] - merged['prob_red_win_gb'])
        disagreements = merged.nlargest(5, 'prob_diff')
        
        for _, row in disagreements.iterrows():
            st.write(f"**{row['RedFighter']} vs {row['BlueFighter']}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"LogReg: {row['predicted_winner_lr']} ({row['prob_red_win_lr']:.1%})")
            with col2:
                st.write(f"GradBoost: {row['predicted_winner_gb']} ({row['prob_red_win_gb']:.1%})")
            st.write(f"Difference: {row['prob_diff']:.1%}")
            st.write("---")
    
    elif page == "Fight Predictions":
        st.header("🥊 Fight Predictions")
        display_fight_cards(logreg_preds, gbdt_preds, upcoming_data)
    
    elif page == "Model Performance":
        st.header("📈 Model Performance")
        
        # Model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            st.write("- ROC-AUC: 66.4%")
            st.write("- Accuracy: 62.1%")
            st.write("- Model: LogisticRegressionCV")
            st.write("- Features: Full pipeline with preprocessing")
            
        with col2:
            st.subheader("Gradient Boosting")
            st.write("- ROC-AUC: 71.1%")
            st.write("- Accuracy: 65.5%")
            st.write("- Model: GradientBoostingClassifier")
            st.write("- Features: Engineered features")
        
        # Feature importance (if available)
        st.subheader("Model Insights")
        st.write("The Gradient Boosting model outperforms Logistic Regression:")
        st.write("- Higher ROC-AUC (71.1% vs 66.4%)")
        st.write("- Better accuracy (65.5% vs 62.1%)")
        st.write("- Generally more confident predictions")
        
        # Training data info
        st.subheader("Training Data")
        st.write("- Historical UFC fight data")
        st.write("- Features include fighter stats, physical attributes, recent form")
        st.write("- Engineered features: win streaks, momentum, physical mismatches")

if __name__ == "__main__":
    main()