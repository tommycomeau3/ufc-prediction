"""
Pytest configuration and shared fixtures for UFC Fight Prediction tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_fight_data():
    """Create sample fight data for testing."""
    return pd.DataFrame({
        'RedFighter': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D', 'Fighter E', 'Fighter F', 'Fighter G', 'Fighter H', 'Fighter I', 'Fighter J'],
        'BlueFighter': ['Fighter X', 'Fighter Y', 'Fighter Z', 'Fighter W', 'Fighter V', 'Fighter U', 'Fighter T', 'Fighter S', 'Fighter R', 'Fighter Q'],
        'Winner': ['Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue'],  # Balanced: 5 Red, 5 Blue
        'Date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01']),
        'RedAge': [28, 30, 25, 29, 27, 31, 26, 32, 28, 30],
        'BlueAge': [26, 32, 27, 28, 30, 29, 31, 25, 29, 27],
        'RedHeight': [180.0, 175.0, 185.0, 178.0, 182.0, 177.0, 183.0, 176.0, 181.0, 179.0],
        'BlueHeight': [178.0, 177.0, 183.0, 180.0, 179.0, 181.0, 182.0, 175.0, 184.0, 178.0],
        'RedReach': [190.0, 180.0, 195.0, 185.0, 192.0, 188.0, 194.0, 186.0, 191.0, 189.0],
        'BlueReach': [185.0, 182.0, 190.0, 187.0, 189.0, 191.0, 193.0, 184.0, 192.0, 188.0],
        'RedWeight': [170.0, 155.0, 185.0, 165.0, 175.0, 160.0, 180.0, 162.0, 172.0, 168.0],
        'BlueWeight': [168.0, 160.0, 180.0, 170.0, 172.0, 162.0, 178.0, 158.0, 174.0, 166.0],
        'RedAvgSigStrLanded': [4.5, 3.2, 5.1, 4.2, 4.8, 3.9, 5.0, 3.5, 4.6, 4.1],
        'BlueAvgSigStrLanded': [4.0, 3.5, 4.8, 4.1, 4.3, 3.7, 4.9, 3.3, 4.4, 3.8],
        'RedAvgSigStrPct': [45.0, 38.0, 52.0, 43.0, 48.0, 40.0, 51.0, 39.0, 47.0, 42.0],
        'BlueAvgSigStrPct': [42.0, 40.0, 48.0, 44.0, 46.0, 41.0, 49.0, 37.0, 45.0, 43.0],
        'RedAvgTDPct': [50.0, 30.0, 60.0, 45.0, 55.0, 35.0, 58.0, 32.0, 53.0, 38.0],
        'BlueAvgTDPct': [45.0, 35.0, 55.0, 50.0, 48.0, 40.0, 57.0, 33.0, 52.0, 36.0],
        'WeightClass': ['Middleweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Welterweight', 'Lightweight', 'Middleweight', 'Lightweight', 'Welterweight', 'Middleweight'],
        'Stance': ['Orthodox', 'Southpaw', 'Orthodox', 'Southpaw', 'Orthodox', 'Southpaw', 'Orthodox', 'Southpaw', 'Orthodox', 'Southpaw'],
    })


@pytest.fixture
def sample_fight_data_with_missing():
    """Create sample fight data with missing values for testing."""
    df = pd.DataFrame({
        'RedFighter': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D', 'Fighter E', 'Fighter F', 'Fighter G', 'Fighter H', 'Fighter I', 'Fighter J'],
        'BlueFighter': ['Fighter X', 'Fighter Y', 'Fighter Z', 'Fighter W', 'Fighter V', 'Fighter U', 'Fighter T', 'Fighter S', 'Fighter R', 'Fighter Q'],
        'Winner': ['Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue'],
        'Date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01']),
        'RedAge': [28, np.nan, 25, 29, 27, np.nan, 26, 32, 28, 30],
        'BlueAge': [26, 32, np.nan, 28, 30, 29, 31, 25, 29, 27],
        'RedHeight': [180.0, 175.0, np.nan, 178.0, 182.0, 177.0, 183.0, 176.0, 181.0, 179.0],
        'BlueHeight': [178.0, np.nan, 183.0, 180.0, 179.0, 181.0, 182.0, 175.0, 184.0, 178.0],
        'RedReach': [190.0, 180.0, 195.0, 185.0, 192.0, 188.0, 194.0, 186.0, 191.0, 189.0],
        'BlueReach': [185.0, 182.0, 190.0, 187.0, 189.0, 191.0, 193.0, 184.0, 192.0, 188.0],
        'RedWeight': [170.0, 155.0, 185.0, 165.0, 175.0, 160.0, 180.0, 162.0, 172.0, 168.0],
        'BlueWeight': [168.0, 160.0, 180.0, 170.0, 172.0, 162.0, 178.0, 158.0, 174.0, 166.0],
        'RedAvgSigStrLanded': [4.5, 3.2, 5.1, 4.2, 4.8, 3.9, 5.0, 3.5, 4.6, 4.1],
        'BlueAvgSigStrLanded': [4.0, 3.5, 4.8, 4.1, 4.3, 3.7, 4.9, 3.3, 4.4, 3.8],
        'RedAvgSigStrPct': [45.0, 38.0, 52.0, 43.0, 48.0, 40.0, 51.0, 39.0, 47.0, 42.0],
        'BlueAvgSigStrPct': [42.0, 40.0, 48.0, 44.0, 46.0, 41.0, 49.0, 37.0, 45.0, 43.0],
        'RedAvgTDPct': [50.0, 30.0, 60.0, 45.0, 55.0, 35.0, 58.0, 32.0, 53.0, 38.0],
        'BlueAvgTDPct': [45.0, 35.0, 55.0, 50.0, 48.0, 40.0, 57.0, 33.0, 52.0, 36.0],
        'WeightClass': ['Middleweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Welterweight', 'Lightweight', 'Middleweight', 'Lightweight', 'Welterweight', 'Middleweight'],
        'Stance': ['Orthodox', 'Southpaw', 'Orthodox', 'Southpaw', 'Orthodox', 'Southpaw', 'Orthodox', 'Southpaw', 'Orthodox', 'Southpaw'],
    })
    return df


@pytest.fixture
def sample_fight_data_high_missing():
    """Create sample data with high missing values (>70%) for testing drop logic."""
    df = pd.DataFrame({
        'RedFighter': ['Fighter A', 'Fighter B'],
        'BlueFighter': ['Fighter X', 'Fighter Y'],
        'Winner': ['Red', 'Blue'],
        'Date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'RedAge': [28, 30],
        'BlueAge': [26, 32],
        'HighMissingCol': [np.nan, np.nan],  # 100% missing
        'AnotherHighMissing': [1.0, np.nan],  # 50% missing (should keep)
    })
    return df


@pytest.fixture
def sample_fighter_stats():
    """Create sample fighter statistics for matchup testing."""
    return pd.DataFrame({
        'fighter': ['Fighter A', 'Fighter B', 'Fighter X', 'Fighter Y'],
        'Age': [28, 30, 26, 32],
        'Height': [180.0, 175.0, 178.0, 177.0],
        'Reach': [190.0, 180.0, 185.0, 182.0],
        'Weight': [170.0, 155.0, 168.0, 160.0],
        'AvgSigStrLanded': [4.5, 3.2, 4.0, 3.5],
        'AvgSigStrPct': [45.0, 38.0, 42.0, 40.0],
        'AvgTDPct': [50.0, 30.0, 45.0, 35.0],
    }).set_index('fighter')


@pytest.fixture
def mock_model_pipeline():
    """Create a mock model pipeline for testing."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=100))
    ])
    return pipeline


@pytest.fixture
def trained_logistic_model(sample_fight_data):
    """Create a trained logistic regression model for testing."""
    from scripts.preprocess import _create_target
    from scripts.features import engineer_features
    
    # Prepare data
    df = _create_target(sample_fight_data.copy())
    df = engineer_features(df)
    
    # Simple train/test split
    X = df.drop(columns=['target', 'Winner', 'RedFighter', 'BlueFighter', 'Date'])
    y = df['target']
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)  # Simple imputation for testing
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    return model, X.columns.tolist()


@pytest.fixture
def trained_gbdt_model(sample_fight_data):
    """Create a trained gradient boosting model for testing."""
    from scripts.preprocess import _create_target
    from scripts.features import engineer_features
    
    # Prepare data
    df = _create_target(sample_fight_data.copy())
    df = engineer_features(df)
    
    # Simple train/test split
    X = df.drop(columns=['target', 'Winner', 'RedFighter', 'BlueFighter', 'Date'])
    y = df['target']
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)  # Simple imputation for testing
    
    model = GradientBoostingClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)
    
    return model, X.columns.tolist()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for test model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_fight_data):
    """Create a temporary CSV file for testing."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_fight_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_model_file(temp_model_dir, trained_logistic_model):
    """Create a temporary model file for testing."""
    model, _ = trained_logistic_model
    model_path = temp_model_dir / "test_model.pkl"
    joblib.dump(model, model_path)
    return model_path

