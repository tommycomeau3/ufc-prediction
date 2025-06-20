# UFC Fight Prediction System

A comprehensive machine learning system that predicts UFC fight outcomes using historical data and advanced fighter statistics. The system features two complementary ML models and an interactive web dashboard for visualization and analysis.

## 🚀 Features

- **Dual ML Models**: Logistic Regression and Gradient Boosting classifiers
- **Advanced Feature Engineering**: Win streaks, momentum indicators, physical mismatches, recent form analysis
- **Interactive Web Dashboard**: Real-time predictions with Plotly visualizations
- **Command-line Tools**: Flexible prediction scripts with customizable parameters
- **Model Comparison**: Side-by-side analysis of prediction differences and agreements

## 📊 Model Performance

- **Gradient Boosting**: 71.1% ROC-AUC, 65.5% Accuracy
- **Logistic Regression**: 66.4% ROC-AUC, 62.1% Accuracy
- **Model Agreement**: 84.6% on upcoming fights

## 🚀 Quick Start - Web Interfaces

### 🥊 Interactive Fighter Predictor (Recommended)
**Enter any two fighters and get instant predictions!**
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive predictor
streamlit run fight_predictor_app.py
```
**Opens at `http://localhost:8501` - Select any two fighters from dropdowns**

### 📊 Predictions Dashboard
**View pre-generated predictions for upcoming fights**
```bash
# Install dependencies
pip install -r requirements.txt

# Generate predictions first (required)
python -m scripts.predict_upcoming
python -m scripts.predict_with_best_model

# Launch the dashboard
streamlit run app.py
```
**Opens at `http://localhost:8501` - Shows analysis of upcoming fight predictions**

## 🏃‍♂️ Detailed Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Predictions (Optional)
```bash
# Generate Logistic Regression predictions
python -m scripts.predict_upcoming

# Generate Gradient Boosting predictions
python -m scripts.predict_with_best_model

# Compare both models
python compare_predictions.py
```

### 3. Launch Web Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🌐 Web Interfaces

### 🥊 Interactive Fighter Predictor (`fight_predictor_app.py`)
**Perfect for exploring "what-if" matchups between any fighters!**

**Features:**
- **Fighter Selection**: Choose any two fighters from dropdown menus with autocomplete
- **Instant Predictions**: Get real-time predictions using both ML models
- **Visual Results**: Beautiful prediction cards with confidence levels
- **Model Comparison**: See how Logistic Regression vs Gradient Boosting compare
- **Fighter Stats**: View historical performance data for selected fighters
- **Fight Analysis**: Detailed breakdown of fighter strengths and matchup dynamics

**How to Use:**
1. Run `streamlit run fight_predictor_app.py`
2. Select Red Corner fighter from dropdown
3. Select Blue Corner fighter from dropdown
4. Click "🥊 PREDICT FIGHT OUTCOME"
5. View prediction results and analysis

### 📊 Predictions Dashboard (`app.py`)
**Analyze pre-generated predictions for upcoming UFC events**

**Features:**
- **Overview Page**: Key metrics, model agreement rates, confidence charts
- **Model Comparison**: Probability scatter plots, disagreement analysis
- **Fight Predictions**: Individual fight cards with detailed predictions
- **Model Performance**: Training metrics, feature engineering insights

**How to Use:**
1. Generate predictions: `python -m scripts.predict_upcoming` and `python -m scripts.predict_with_best_model`
2. Run `streamlit run app.py`
3. Navigate between pages using the sidebar
4. Explore model comparisons and fight analysis

## 📁 Project Structure

```
├── fight_predictor_app.py          # 🥊 Interactive fighter predictor (MAIN APP)
├── app.py                          # 📊 Predictions dashboard for upcoming fights
├── main.py                         # Logistic regression training pipeline
├── compare_predictions.py          # Model comparison utility
├── requirements.txt                # Python dependencies
├── data/
│   ├── ufc-master.csv             # Historical UFC fight data
│   ├── upcoming.csv               # Upcoming fights to predict
│   ├── fighter_stats.csv          # Fighter statistics cache
│   └── upcoming_with_preds.csv    # Upcoming fights with predictions
├── models/
│   ├── best.pkl                   # Best performing model (Gradient Boosting)
│   ├── stage1_logreg_pipeline.pkl # Logistic regression pipeline
│   └── [other model files]        # Additional trained models
├── scripts/
│   ├── features.py                # Feature engineering functions
│   ├── matchup.py                 # Fighter matchup utilities
│   ├── model.py                   # Model training utilities
│   ├── preprocess.py              # Data preprocessing pipeline
│   ├── train.py                   # Training orchestrator
│   ├── predict_upcoming.py        # LogReg prediction script
│   └── predict_with_best_model.py # GradBoost prediction script
├── predictions.csv                # Logistic regression predictions
├── predictions_gbdt.csv           # Gradient boosting predictions
└── docs/                          # Documentation files
```

## 🛠️ Advanced Usage

### Command Line Predictions

#### Custom Data Sources
```bash
# Use custom upcoming fights data
python -m scripts.predict_upcoming --data my_fights.csv --output my_predictions.csv

# Use different trained model
python -m scripts.predict_with_best_model --model models/custom_model.pkl
```

#### Training New Models
```bash
# Train gradient boosting model
python -m scripts.train --model gbdt

# Train logistic regression model  
python -m scripts.train --model logreg

# Train with custom parameters
python main.py  # For logistic regression with full pipeline
```

### Model Training Pipeline

The training process involves:
1. **Data Loading**: Historical UFC fight data preprocessing
2. **Feature Engineering**: Creating differential and momentum features
3. **Model Training**: Cross-validated training with hyperparameter tuning
4. **Model Evaluation**: Performance metrics on held-out test set
5. **Model Persistence**: Saving trained models for inference

## 🧠 Feature Engineering

The system employs sophisticated feature engineering:

### Core Features
- **Fighter Statistics**: Win/loss records, finishing rates, decision rates
- **Physical Attributes**: Height, weight, reach, age differences
- **Performance Metrics**: Striking accuracy, takedown success, defense rates

### Advanced Features
- **Differential Features**: Red vs Blue fighter statistical comparisons
- **Momentum Indicators**: Win streaks, recent form, time since last fight
- **Physical Mismatches**: Height, reach, weight, and age advantages
- **Performance Ratios**: Striking, grappling, and defensive comparisons
- **Rolling Averages**: Time-weighted performance trends over recent fights

### Feature Selection
- Automated high-missing value column removal (>70% threshold)
- Correlation-based feature filtering
- Cross-validation based feature importance ranking

## 🎯 Prediction Workflow

1. **Data Preparation**: Load and clean upcoming fight data
2. **Feature Engineering**: Apply same transformations as training data
3. **Model Inference**: Generate predictions using both models
4. **Post-processing**: Calculate confidence scores and winner predictions
5. **Output Generation**: Save predictions to CSV files
6. **Visualization**: Display results in web dashboard

## 📈 Model Details

### Logistic Regression Pipeline
- **Algorithm**: LogisticRegressionCV with 5-fold cross-validation
- **Preprocessing**: StandardScaler + OneHotEncoder
- **Regularization**: L2 penalty with automatic C selection
- **Class Balancing**: Balanced class weights

### Gradient Boosting Classifier
- **Algorithm**: GradientBoostingClassifier
- **Features**: Engineered feature set with momentum indicators
- **Hyperparameters**: Optimized through grid search
- **Validation**: Stratified cross-validation

## 🔧 Configuration

### Environment Requirements
- Python 3.8+
- Streamlit for web interface
- Scikit-learn for ML models
- Pandas/NumPy for data processing
- Plotly for interactive visualizations

### Data Requirements
- Historical UFC fight data in CSV format
- Upcoming fights data with fighter names and basic info
- Consistent column naming convention

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage policies and applicable regulations when using UFC fight data.

## 🆘 Troubleshooting

### Common Issues

**Prediction files not found**: Run the prediction scripts before launching the dashboard
```bash
python -m scripts.predict_upcoming
python -m scripts.predict_with_best_model
streamlit run app.py
```

**Model loading errors**: Ensure model files exist in the `models/` directory
**Feature mismatch errors**: Verify that upcoming data has consistent column structure with training data
**Memory issues**: Consider reducing feature set size or using feature selection

### Support

For issues and questions, please check the documentation in the `docs/` directory or create an issue in the repository.
