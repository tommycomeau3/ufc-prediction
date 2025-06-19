# UFC Fight Prediction System

A machine learning system that predicts UFC fight outcomes using historical data and fighter statistics.

## Features

- **Two ML Models**: Logistic Regression and Gradient Boosting
- **Feature Engineering**: Win streaks, momentum, physical mismatches, recent form
- **Web Dashboard**: Interactive visualization of predictions and model performance
- **Prediction Scripts**: Command-line tools for generating predictions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Predictions
```bash
# Logistic Regression predictions
python -m scripts.predict_upcoming

# Gradient Boosting predictions  
python -m scripts.predict_with_best_model

# Compare both models
python compare_predictions.py
```

### 3. Launch Web Dashboard
```bash
streamlit run app.py
```

## Model Performance

- **Gradient Boosting**: 71.1% ROC-AUC, 65.5% Accuracy
- **Logistic Regression**: 66.4% ROC-AUC, 62.1% Accuracy
- **Model Agreement**: 84.6% on upcoming fights

## Dashboard Features

### 📊 Overview
- Key metrics and model agreement rates
- Confidence comparison charts
- Model performance summary

### 🔍 Model Comparison  
- Probability scatter plots
- Disagreement analysis
- Side-by-side predictions

### 🥊 Fight Predictions
- Individual fight cards with predictions
- Confidence levels for each model
- Agreement indicators

### 📈 Model Performance
- Training metrics and model details
- Feature engineering insights
- Data pipeline information

## File Structure

```
├── app.py                          # Streamlit web dashboard
├── main.py                         # Original training pipeline
├── compare_predictions.py          # Model comparison script
├── requirements.txt                # Python dependencies
├── data/
│   ├── ufc-master.csv             # Historical training data
│   └── upcoming.csv               # Upcoming fights to predict
├── models/
│   ├── best.pkl                   # Best performing model (Gradient Boosting)
│   ├── stage1_logreg_pipeline.pkl # Logistic regression pipeline
│   └── ...
├── scripts/
│   ├── features.py                # Feature engineering
│   ├── model.py                   # Model training utilities
│   ├── preprocess.py              # Data preprocessing
│   ├── train.py                   # Training orchestrator
│   ├── predict_upcoming.py        # LogReg predictions
│   └── predict_with_best_model.py # GradBoost predictions
└── predictions.csv                # Generated predictions
```

## Usage Examples

### Command Line Predictions
```bash
# Generate predictions with custom data
python -m scripts.predict_upcoming --data my_fights.csv --output my_predictions.csv

# Use different model
python -m scripts.predict_with_best_model --model models/gbdt.pkl
```

### Training New Models
```bash
# Train gradient boosting model
python -m scripts.train --model gbdt

# Train logistic regression model  
python -m scripts.train --model logreg
```

## Model Features

The system uses comprehensive feature engineering including:

- **Differential Features**: Red vs Blue fighter comparisons
- **Momentum Features**: Win streaks and recent form
- **Physical Mismatches**: Height, reach, weight, age differences
- **Performance Ratios**: Striking, takedown, and defense comparisons
- **Rolling Averages**: Time-weighted performance trends

## Web Dashboard

The Streamlit dashboard provides:
- Interactive visualizations with Plotly
- Real-time model comparisons
- Detailed fight-by-fight analysis
- Model performance metrics
- Responsive design for desktop and mobile

Launch with: `streamlit run app.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.
