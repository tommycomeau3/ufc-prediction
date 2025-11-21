"""
UFC Fight Prediction Dashboard - Redesigned

A modern, beautiful Streamlit web application for visualizing UFC fight predictions and model performance.

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
    page_title="UFC Fight Predictor | Dashboard",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
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
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    
    /* Fight Card */
    .fight-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .fight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF0000 0%, #000000 50%, #0066FF 100%);
    }
    
    .fight-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.08);
    }
    
    .fighter-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    
    .fighter-name.red {
        color: #DC2626;
    }
    
    .fighter-name.blue {
        color: #2563EB;
    }
    
    .vs-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #6b7280;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .prediction-badge.winner {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
    }
    
    .prediction-badge.loser {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e5e7eb;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
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
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #FF0000 0%, #000000 50%, #0066FF 100%) 1;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Agreement Indicator */
    .agreement-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .agreement-indicator.agree {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .agreement-indicator.disagree {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        text-align: center;
        padding: 0.75rem;
        background: #f9fafb;
        border-radius: 8px;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.25rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        st.error("⚠️ Prediction files not found. Please run the prediction scripts first.")
        return None, None

@st.cache_data
def load_upcoming_data():
    """Load upcoming fights data."""
    try:
        return pd.read_csv("data/upcoming.csv")
    except FileNotFoundError:
        return None

def create_confidence_chart(logreg_preds, gbdt_preds):
    """Create an enhanced confidence comparison chart."""
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    merged['fight'] = merged['RedFighter'] + ' vs ' + merged['BlueFighter']
    merged = merged.sort_values('confidence_gbdt', ascending=True)
    
    fig = go.Figure()
    
    # Add LogReg confidence
    fig.add_trace(go.Bar(
        name='Logistic Regression',
        x=merged['confidence_logreg'],
        y=merged['fight'],
        orientation='h',
        marker=dict(
            color='#DC2626',
            line=dict(color='#991B1B', width=1)
        ),
        text=[f"{c:.1%}" for c in merged['confidence_logreg']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Confidence: %{text}<extra></extra>'
    ))
    
    # Add GradBoost confidence
    fig.add_trace(go.Bar(
        name='Gradient Boosting',
        x=merged['confidence_gbdt'],
        y=merged['fight'],
        orientation='h',
        marker=dict(
            color='#2563EB',
            line=dict(color='#1E40AF', width=1)
        ),
        text=[f"{c:.1%}" for c in merged['confidence_gbdt']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Confidence: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Model Confidence Comparison</b>",
            font=dict(size=20, color='#111827')
        ),
        xaxis=dict(
            title="Confidence Level",
            tickformat='.0%',
            gridcolor='#e5e7eb'
        ),
        yaxis=dict(
            title="",
            gridcolor='#e5e7eb'
        ),
        barmode='group',
        height=max(400, len(merged) * 50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=200, r=20, t=80, b=40)
    )
    
    return fig

def create_agreement_chart(logreg_preds, gbdt_preds):
    """Create an enhanced model agreement visualization."""
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    merged['agrees'] = merged['predicted_winner_logreg'] == merged['predicted_winner_gbdt']
    agreement_rate = merged['agrees'].mean()
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Models Agree', 'Models Disagree'],
        values=[merged['agrees'].sum(), (~merged['agrees']).sum()],
        hole=0.6,
        marker=dict(
            colors=['#10B981', '#EF4444'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=14, family='Inter', color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>Model Agreement Rate: {agreement_rate:.1%}</b>",
            font=dict(size=18, color='#111827'),
            x=0.5
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        annotations=[dict(
            text=f'<b>{agreement_rate:.1%}</b>',
            x=0.5, y=0.5,
            font=dict(size=32, color='#111827', family='Inter'),
            showarrow=False
        )]
    )
    
    return fig

def create_probability_scatter(logreg_preds, gbdt_preds):
    """Create an enhanced scatter plot comparing probabilities."""
    merged = pd.merge(
        logreg_preds, gbdt_preds, 
        on=["RedFighter", "BlueFighter"], 
        suffixes=("_logreg", "_gbdt")
    )
    
    merged['fight'] = merged['RedFighter'] + ' vs ' + merged['BlueFighter']
    merged['agrees'] = merged['predicted_winner_logreg'] == merged['predicted_winner_gbdt']
    
    fig = go.Figure()
    
    # Add scatter points for agreeing predictions
    agrees = merged[merged['agrees']]
    if len(agrees) > 0:
        fig.add_trace(go.Scatter(
            x=agrees['prob_red_win_logreg'],
            y=agrees['prob_red_win_gbdt'],
            mode='markers',
            name='Models Agree',
            marker=dict(
                size=12,
                color='#10B981',
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=agrees['fight'],
            hovertemplate='<b>%{text}</b><br>LogReg: %{x:.1%}<br>GradBoost: %{y:.1%}<extra></extra>'
        ))
    
    # Add scatter points for disagreeing predictions
    disagrees = merged[~merged['agrees']]
    if len(disagrees) > 0:
        fig.add_trace(go.Scatter(
            x=disagrees['prob_red_win_logreg'],
            y=disagrees['prob_red_win_gbdt'],
            mode='markers',
            name='Models Disagree',
            marker=dict(
                size=12,
                color='#EF4444',
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=disagrees['fight'],
            hovertemplate='<b>%{text}</b><br>LogReg: %{x:.1%}<br>GradBoost: %{y:.1%}<extra></extra>'
        ))
    
    # Add diagonal line (perfect agreement)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='#6b7280', width=2),
        name='Perfect Agreement',
        hovertemplate=''
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Red Win Probability Comparison</b>",
            font=dict(size=20, color='#111827')
        ),
        xaxis=dict(
            title="Logistic Regression Probability",
            tickformat='.0%',
            gridcolor='#e5e7eb',
            range=[0, 1]
        ),
        yaxis=dict(
            title="Gradient Boosting Probability",
            tickformat='.0%',
            gridcolor='#e5e7eb',
            range=[0, 1]
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_confidence_distribution(logreg_preds, gbdt_preds):
    """Create a distribution chart of confidence levels."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=logreg_preds['confidence'],
        name='Logistic Regression',
        nbinsx=20,
        marker=dict(
            color='#DC2626',
            line=dict(color='#991B1B', width=1)
        ),
        opacity=0.7
    ))
    
    fig.add_trace(go.Histogram(
        x=gbdt_preds['confidence'],
        name='Gradient Boosting',
        nbinsx=20,
        marker=dict(
            color='#2563EB',
            line=dict(color='#1E40AF', width=1)
        ),
        opacity=0.7
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Confidence Distribution</b>",
            font=dict(size=18, color='#111827')
        ),
        xaxis=dict(
            title="Confidence Level",
            tickformat='.0%',
            gridcolor='#e5e7eb'
        ),
        yaxis=dict(
            title="Number of Fights",
            gridcolor='#e5e7eb'
        ),
        barmode='overlay',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def display_fight_cards(logreg_preds, gbdt_preds, upcoming_data):
    """Display enhanced individual fight prediction cards."""
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
    
    for idx, fight in merged.iterrows():
        # Determine agreement
        agrees = fight['predicted_winner_logreg'] == fight['predicted_winner_gbdt']
        
        # Create fight card HTML
        st.markdown(f'''
        <div class="fight-card">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1.5rem;">
                <div>
                    <div class="fighter-name red">🔴 {fight['RedFighter']}</div>
                    <div style="color: #6b7280; font-size: 0.875rem;">
                        {fight.get('WeightClass', 'N/A')} • {fight.get('Date', 'N/A')}
                    </div>
                </div>
                <div class="vs-text">VS</div>
                <div style="text-align: right;">
                    <div class="fighter-name blue">🔵 {fight['BlueFighter']}</div>
                    <div style="color: #6b7280; font-size: 0.875rem;">
                        {fight.get('WeightClass', 'N/A')} • {fight.get('Date', 'N/A')}
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;">
                <div style="padding: 1rem; background: #f9fafb; border-radius: 12px;">
                    <div style="font-weight: 600; color: #6b7280; margin-bottom: 0.75rem; font-size: 0.875rem;">
                        LOGISTIC REGRESSION
                    </div>
                    <div style="margin-bottom: 0.5rem;">
                        <span class="prediction-badge {'winner' if fight['predicted_winner_logreg'] == 'Red' else 'loser'}">
                            🔴 {fight['RedFighter']} ({fight['prob_red_win_logreg']:.1%})
                        </span>
                        <span class="prediction-badge {'winner' if fight['predicted_winner_logreg'] == 'Blue' else 'loser'}" style="margin-left: 0.5rem;">
                            🔵 {fight['BlueFighter']} ({1-fight['prob_red_win_logreg']:.1%})
                        </span>
                    </div>
                    <div style="font-size: 0.875rem; color: #6b7280;">
                        Confidence: <strong>{fight['confidence_logreg']:.1%}</strong>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill {'high' if fight['confidence_logreg'] > 0.7 else 'medium' if fight['confidence_logreg'] > 0.55 else 'low'}" 
                             style="width: {fight['confidence_logreg']*100}%"></div>
                    </div>
                </div>
                
                <div style="padding: 1rem; background: #f9fafb; border-radius: 12px;">
                    <div style="font-weight: 600; color: #6b7280; margin-bottom: 0.75rem; font-size: 0.875rem;">
                        GRADIENT BOOSTING
                    </div>
                    <div style="margin-bottom: 0.5rem;">
                        <span class="prediction-badge {'winner' if fight['predicted_winner_gbdt'] == 'Red' else 'loser'}">
                            🔴 {fight['RedFighter']} ({fight['prob_red_win_gbdt']:.1%})
                        </span>
                        <span class="prediction-badge {'winner' if fight['predicted_winner_gbdt'] == 'Blue' else 'loser'}" style="margin-left: 0.5rem;">
                            🔵 {fight['BlueFighter']} ({1-fight['prob_red_win_gbdt']:.1%})
                        </span>
                    </div>
                    <div style="font-size: 0.875rem; color: #6b7280;">
                        Confidence: <strong>{fight['confidence_gbdt']:.1%}</strong>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill {'high' if fight['confidence_gbdt'] > 0.7 else 'medium' if fight['confidence_gbdt'] > 0.55 else 'low'}" 
                             style="width: {fight['confidence_gbdt']*100}%"></div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1rem; text-align: center;">
                <span class="agreement-indicator {'agree' if agrees else 'disagree'}">
                    {'✅ Models Agree' if agrees else '❌ Models Disagree'}
                </span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🥊 UFC Fight Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning Predictions & Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Load data
    logreg_preds, gbdt_preds = load_predictions()
    upcoming_data = load_upcoming_data()
    
    if logreg_preds is None or gbdt_preds is None:
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["📊 Overview", "🔍 Model Comparison", "🥊 Fight Predictions", "📈 Model Performance"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    
    # Calculate quick stats
    merged = pd.merge(logreg_preds, gbdt_preds, on=["RedFighter", "BlueFighter"], suffixes=("_lr", "_gb"))
    agreement = (merged['predicted_winner_lr'] == merged['predicted_winner_gb']).mean()
    avg_conf_lr = logreg_preds['confidence'].mean()
    avg_conf_gb = gbdt_preds['confidence'].mean()
    
    st.sidebar.metric("Total Fights", len(logreg_preds))
    st.sidebar.metric("Model Agreement", f"{agreement:.1%}")
    st.sidebar.metric("Avg LogReg Confidence", f"{avg_conf_lr:.1%}")
    st.sidebar.metric("Avg GradBoost Confidence", f"{avg_conf_gb:.1%}")
    
    # Page Content
    if page == "📊 Overview":
        st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
        
        # Key metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-label">Total Fights</div>
                <div class="metric-value">{len(logreg_preds)}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-label">Model Agreement</div>
                <div class="metric-value">{agreement:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col3:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-label">Avg LogReg Confidence</div>
                <div class="metric-value">{avg_conf_lr:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col4:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-label">Avg GradBoost Confidence</div>
                <div class="metric-value">{avg_conf_gb:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_agreement = create_agreement_chart(logreg_preds, gbdt_preds)
            st.plotly_chart(fig_agreement, use_container_width=True)
            
        with col2:
            fig_distribution = create_confidence_distribution(logreg_preds, gbdt_preds)
            st.plotly_chart(fig_distribution, use_container_width=True)
        
        # Confidence comparison
        st.markdown('<div class="section-header">Confidence Comparison by Fight</div>', unsafe_allow_html=True)
        fig_confidence = create_confidence_chart(logreg_preds, gbdt_preds)
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    elif page == "🔍 Model Comparison":
        st.markdown('<div class="section-header">Model Comparison Analysis</div>', unsafe_allow_html=True)
        
        # Probability scatter plot
        fig_scatter = create_probability_scatter(logreg_preds, gbdt_preds)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Disagreement analysis
        st.markdown('<div class="section-header">Biggest Disagreements</div>', unsafe_allow_html=True)
        merged = pd.merge(logreg_preds, gbdt_preds, on=["RedFighter", "BlueFighter"], suffixes=("_lr", "_gb"))
        merged['prob_diff'] = abs(merged['prob_red_win_lr'] - merged['prob_red_win_gb'])
        merged['agrees'] = merged['predicted_winner_lr'] == merged['predicted_winner_gb']
        disagreements = merged.nlargest(10, 'prob_diff')
        
        for idx, row in disagreements.iterrows():
            agrees = row['agrees']
            st.markdown(f'''
            <div class="fight-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #111827;">{row['RedFighter']} vs {row['BlueFighter']}</h3>
                    <span class="agreement-indicator {'agree' if agrees else 'disagree'}">
                        {'✅ Agree' if agrees else '❌ Disagree'}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <div style="font-weight: 600; color: #6b7280; margin-bottom: 0.5rem;">Logistic Regression</div>
                        <div>Winner: <strong>{row['predicted_winner_lr']}</strong> ({row['prob_red_win_lr']:.1%})</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #6b7280; margin-bottom: 0.5rem;">Gradient Boosting</div>
                        <div>Winner: <strong>{row['predicted_winner_gb']}</strong> ({row['prob_red_win_gb']:.1%})</div>
                    </div>
                </div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
                    <div style="color: #6b7280;">Probability Difference: <strong>{row['prob_diff']:.1%}</strong></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    elif page == "🥊 Fight Predictions":
        st.markdown('<div class="section-header">Upcoming Fight Predictions</div>', unsafe_allow_html=True)
        display_fight_cards(logreg_preds, gbdt_preds, upcoming_data)
    
    elif page == "📈 Model Performance":
        st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        # Model info cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <div class="fight-card">
                <h2 style="color: #DC2626; margin-bottom: 1rem;">Logistic Regression</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">ROC-AUC</div>
                        <div class="stat-value">66.4%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Accuracy</div>
                        <div class="stat-value">62.1%</div>
                    </div>
                </div>
                <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #e5e7eb;">
                    <div style="color: #6b7280; line-height: 1.8;">
                        <strong>Model Type:</strong> LogisticRegressionCV<br>
                        <strong>Features:</strong> Full pipeline with preprocessing<br>
                        <strong>Training:</strong> Historical UFC fight data
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown('''
            <div class="fight-card">
                <h2 style="color: #2563EB; margin-bottom: 1rem;">Gradient Boosting</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">ROC-AUC</div>
                        <div class="stat-value">71.1%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Accuracy</div>
                        <div class="stat-value">65.5%</div>
                    </div>
                </div>
                <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #e5e7eb;">
                    <div style="color: #6b7280; line-height: 1.8;">
                        <strong>Model Type:</strong> GradientBoostingClassifier<br>
                        <strong>Features:</strong> Engineered features<br>
                        <strong>Training:</strong> Historical UFC fight data
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Model insights
        st.markdown('<div class="section-header">Model Insights</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="fight-card">
            <h3 style="color: #111827; margin-bottom: 1rem;">Key Findings</h3>
            <div style="color: #374151; line-height: 1.8;">
                <p><strong>🏆 Gradient Boosting Outperforms:</strong></p>
                <ul style="margin-left: 1.5rem;">
                    <li>Higher ROC-AUC (71.1% vs 66.4%) - Better at distinguishing winners</li>
                    <li>Better accuracy (65.5% vs 62.1%) - More correct predictions</li>
                    <li>Generally more confident predictions across fights</li>
                </ul>
                
                <p style="margin-top: 1.5rem;"><strong>📊 Training Data:</strong></p>
                <ul style="margin-left: 1.5rem;">
                    <li>Historical UFC fight data with comprehensive fighter statistics</li>
                    <li>Features include: fighter stats, physical attributes, recent form, win streaks</li>
                    <li>Engineered features: momentum indicators, physical mismatches, experience differentials</li>
                </ul>
                
                <p style="margin-top: 1.5rem;"><strong>💡 Model Agreement:</strong></p>
                <ul style="margin-left: 1.5rem;">
                    <li>Models agree on {:.1%} of predictions</li>
                    <li>Disagreements often occur in close matchups with similar fighter profiles</li>
                    <li>Higher confidence predictions show better agreement between models</li>
                </ul>
            </div>
        </div>
        '''.format(agreement), unsafe_allow_html=True)
    
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
