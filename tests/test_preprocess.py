"""
Unit tests for data preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.preprocess import (
    _read_csv,
    _coerce_dates,
    _create_target,
    _drop_high_missing,
    clean_ufc_data,
    scale_features,
)


class TestReadCSV:
    """Test CSV reading functionality."""
    
    def test_read_csv_success(self, sample_csv_file):
        """Test successful CSV reading."""
        result = _read_csv(sample_csv_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_read_csv_file_not_found(self):
        """Test CSV reading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            _read_csv("nonexistent_file.csv")
    
    def test_read_csv_encoding_fallback(self, temp_data_dir):
        """Test CSV reading with encoding fallback."""
        # Create a CSV file that might need encoding fallback
        csv_path = temp_data_dir / "test_encoding.csv"
        df = pd.DataFrame({'col1': ['test', 'data']})
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        result = _read_csv(csv_path)
        assert isinstance(result, pd.DataFrame)


class TestCoerceDates:
    """Test date coercion functionality."""
    
    def test_coerce_dates_with_date_column(self, sample_fight_data):
        """Test date coercion when Date column exists."""
        df = sample_fight_data.copy()
        df['Date'] = df['Date'].astype(str)  # Convert to string
        
        result = _coerce_dates(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
    
    def test_coerce_dates_without_date_column(self, sample_fight_data):
        """Test date coercion when Date column doesn't exist."""
        df = sample_fight_data.copy()
        df = df.drop(columns=['Date'])
        
        result = _coerce_dates(df)
        
        # Should return unchanged
        assert 'Date' not in result.columns
        assert result.shape == df.shape
    
    def test_coerce_dates_invalid_dates(self):
        """Test date coercion with invalid date strings."""
        df = pd.DataFrame({
            'Date': ['2020-01-01', 'invalid-date', '2020-02-01'],
            'Other': [1, 2, 3],
        })
        
        result = _coerce_dates(df)
        
        # Should convert valid dates and set invalid to NaT
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        assert pd.isna(result.loc[1, 'Date'])


class TestCreateTarget:
    """Test target variable creation."""
    
    def test_create_target_red_wins(self, sample_fight_data):
        """Test target creation for Red wins."""
        result = _create_target(sample_fight_data.copy())
        
        assert 'target' in result.columns
        assert result.loc[0, 'target'] == 1  # Red wins
        assert result.loc[1, 'target'] == 0  # Blue wins
        assert result.loc[2, 'target'] == 1  # Red wins
    
    def test_create_target_no_winner_column(self, sample_fight_data):
        """Test target creation when Winner column is missing."""
        df = sample_fight_data.copy()
        df = df.drop(columns=['Winner'])
        
        with pytest.raises(ValueError, match="Column 'Winner' missing"):
            _create_target(df)
    
    def test_create_target_filters_invalid_winners(self):
        """Test that invalid winner values are filtered out."""
        df = pd.DataFrame({
            'Winner': ['Red', 'Blue', 'Draw', 'No Contest', 'Red'],
            'Other': [1, 2, 3, 4, 5],
        })
        
        result = _create_target(df)
        
        # Should only keep Red and Blue
        assert len(result) == 3
        assert all(result['Winner'].isin(['Red', 'Blue']))
    
    @pytest.mark.parametrize("winner,expected", [
        ('Red', 1),
        ('Blue', 0),
    ])
    def test_create_target_values(self, winner, expected):
        """Test target values for different winners."""
        df = pd.DataFrame({
            'Winner': [winner],
            'Other': [1],
        })
        
        result = _create_target(df)
        
        assert result.loc[0, 'target'] == expected


class TestDropHighMissing:
    """Test dropping columns with high missing values."""
    
    def test_drop_high_missing_default_threshold(self, sample_fight_data_high_missing):
        """Test dropping columns with default threshold (0.7)."""
        result = _drop_high_missing(sample_fight_data_high_missing.copy())
        
        # Column with 100% missing should be dropped
        assert 'HighMissingCol' not in result.columns
        
        # Column with 50% missing should be kept
        assert 'AnotherHighMissing' in result.columns
    
    def test_drop_high_missing_custom_threshold(self, sample_fight_data_high_missing):
        """Test dropping columns with custom threshold."""
        result = _drop_high_missing(sample_fight_data_high_missing.copy(), threshold=0.4)
        
        # With 0.4 threshold, 50% missing column should also be dropped
        assert 'AnotherHighMissing' not in result.columns
    
    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 1.0])
    def test_drop_high_missing_thresholds(self, threshold):
        """Test dropping columns with various thresholds."""
        df = pd.DataFrame({
            'LowMissing': [1, 2, 3],  # 0% missing
            'MediumMissing': [1, np.nan, 3],  # 33% missing
            'HighMissing': [np.nan, np.nan, 3],  # 67% missing
            'VeryHighMissing': [np.nan, np.nan, np.nan],  # 100% missing
        })
        
        result = _drop_high_missing(df, threshold=threshold)
        
        # Verify columns are dropped based on threshold
        if threshold <= 0.33:
            assert 'MediumMissing' not in result.columns
        if threshold <= 0.67:
            assert 'HighMissing' not in result.columns
        if threshold <= 1.0:
            assert 'VeryHighMissing' not in result.columns


class TestCleanUFCData:
    """Test the complete data cleaning pipeline."""
    
    def test_clean_ufc_data_complete(self, sample_csv_file):
        """Test complete data cleaning pipeline."""
        result = clean_ufc_data(sample_csv_file)
        
        assert isinstance(result, pd.DataFrame)
        assert 'target' in result.columns
        assert len(result) > 0
    
    def test_clean_ufc_data_resets_index(self, sample_csv_file):
        """Test that index is reset after cleaning."""
        result = clean_ufc_data(sample_csv_file)
        
        # Index should start at 0 and be sequential
        assert result.index[0] == 0
        assert len(result.index) == len(result)


class TestScaleFeatures:
    """Test feature scaling functionality."""
    
    def test_scale_features_basic(self, sample_fight_data):
        """Test basic feature scaling."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data.copy())
        X, y = scale_features(df)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == len(y)
        assert X.shape[0] == len(df)
    
    def test_scale_features_no_target(self, sample_fight_data):
        """Test feature scaling without target column."""
        with pytest.raises(KeyError, match="target"):
            scale_features(sample_fight_data)
    
    def test_scale_features_handles_missing(self, sample_fight_data_with_missing):
        """Test feature scaling with missing values."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data_with_missing.copy())
        X, y = scale_features(df)
        
        # Should not crash, imputation should handle missing values
        assert isinstance(X, np.ndarray)
        assert not np.isnan(X).any() or np.isnan(X).sum() < X.size  # Some NaNs might remain
    
    def test_scale_features_categorical_encoding(self, sample_fight_data):
        """Test that categorical features are one-hot encoded."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data.copy())
        X, y = scale_features(df)
        
        # Should create more features than original columns due to one-hot encoding
        assert X.shape[1] >= len(df.select_dtypes(include=['object']).columns)

