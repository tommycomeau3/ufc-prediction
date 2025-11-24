"""
Data validation tests.
"""
import pytest
import pandas as pd
import numpy as np
from scripts.preprocess import clean_ufc_data, _create_target, _drop_high_missing
from scripts.features import engineer_features


class TestDataQuality:
    """Test data quality and validation."""
    
    def test_data_has_required_columns(self, sample_fight_data):
        """Test that data has required columns."""
        required_columns = ['RedFighter', 'BlueFighter', 'Winner']
        
        for col in required_columns:
            assert col in sample_fight_data.columns, f"Missing required column: {col}"
    
    def test_data_has_no_duplicate_rows(self, sample_fight_data):
        """Test that data has no duplicate rows."""
        duplicates = sample_fight_data.duplicated()
        assert not duplicates.any(), "Data contains duplicate rows"
    
    def test_winner_column_values(self, sample_fight_data):
        """Test that Winner column has valid values."""
        valid_winners = ['Red', 'Blue']
        winner_values = sample_fight_data['Winner'].unique()
        
        assert all(w in valid_winners for w in winner_values), \
            f"Invalid winner values found: {set(winner_values) - set(valid_winners)}"
    
    def test_fighter_names_not_empty(self, sample_fight_data):
        """Test that fighter names are not empty."""
        assert not sample_fight_data['RedFighter'].isna().any(), "RedFighter has missing values"
        assert not sample_fight_data['BlueFighter'].isna().any(), "BlueFighter has missing values"
        
        assert (sample_fight_data['RedFighter'].str.strip() != '').all(), \
            "RedFighter has empty strings"
        assert (sample_fight_data['BlueFighter'].str.strip() != '').all(), \
            "BlueFighter has empty strings"
    
    def test_fighters_not_same(self, sample_fight_data):
        """Test that Red and Blue fighters are different."""
        same_fighters = sample_fight_data['RedFighter'] == sample_fight_data['BlueFighter']
        assert not same_fighters.any(), "Some fights have same fighter in both corners"


class TestDataCompleteness:
    """Test data completeness."""
    
    def test_missing_value_threshold(self, sample_fight_data_high_missing):
        """Test that high missing value columns are dropped."""
        result = _drop_high_missing(sample_fight_data_high_missing.copy(), threshold=0.7)
        
        # Check that high missing columns are dropped
        missing_ratios = sample_fight_data_high_missing.isna().mean()
        high_missing_cols = missing_ratios[missing_ratios >= 0.7].index
        
        for col in high_missing_cols:
            assert col not in result.columns, f"High missing column {col} was not dropped"
    
    def test_data_shape_after_cleaning(self, sample_fight_data):
        """Test that data shape is reasonable after cleaning."""
        df_clean = _create_target(sample_fight_data.copy())
        
        # Should have at least some rows
        assert len(df_clean) > 0, "Data has no rows after cleaning"
        
        # Should have target column
        assert 'target' in df_clean.columns, "Target column missing after cleaning"
    
    def test_numeric_columns_are_numeric(self, sample_fight_data):
        """Test that numeric columns are actually numeric."""
        numeric_cols = sample_fight_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_fight_data[col]), \
                f"Column {col} is marked as numeric but isn't"


class TestDataConsistency:
    """Test data consistency."""
    
    def test_date_column_format(self, sample_fight_data):
        """Test that Date column is in correct format."""
        if 'Date' in sample_fight_data.columns:
            # Should be datetime or convertible to datetime
            dates = pd.to_datetime(sample_fight_data['Date'], errors='coerce')
            assert not dates.isna().all(), "Date column cannot be converted to datetime"
    
    def test_age_values_reasonable(self, sample_fight_data):
        """Test that age values are reasonable."""
        if 'RedAge' in sample_fight_data.columns:
            ages = sample_fight_data['RedAge'].dropna()
            if len(ages) > 0:
                assert (ages >= 18).all(), "Some fighters have age < 18"
                assert (ages <= 60).all(), "Some fighters have age > 60"
    
    def test_height_values_reasonable(self, sample_fight_data):
        """Test that height values are reasonable (in cm)."""
        if 'RedHeight' in sample_fight_data.columns:
            heights = sample_fight_data['RedHeight'].dropna()
            if len(heights) > 0:
                # Reasonable height range: 150-220 cm
                assert (heights >= 150).all(), "Some fighters have height < 150 cm"
                assert (heights <= 220).all(), "Some fighters have height > 220 cm"
    
    def test_weight_values_reasonable(self, sample_fight_data):
        """Test that weight values are reasonable (in lbs or kg)."""
        if 'RedWeight' in sample_fight_data.columns:
            weights = sample_fight_data['RedWeight'].dropna()
            if len(weights) > 0:
                # Reasonable weight range: 100-350 lbs (or 45-160 kg)
                assert (weights >= 100).all(), "Some fighters have weight < 100"
                assert (weights <= 350).all(), "Some fighters have weight > 350"


class TestFeatureEngineeringValidation:
    """Test validation of feature engineering output."""
    
    def test_feature_engineering_creates_features(self, sample_fight_data):
        """Test that feature engineering creates new features."""
        original_cols = set(sample_fight_data.columns)
        df_features = engineer_features(sample_fight_data.copy())
        new_cols = set(df_features.columns)
        
        # Should have more columns after feature engineering
        assert len(new_cols) > len(original_cols), \
            "Feature engineering did not create new features"
    
    def test_differential_features_correct(self, sample_fight_data):
        """Test that differential features are calculated correctly."""
        df_features = engineer_features(sample_fight_data.copy())
        
        if 'Age_diff' in df_features.columns and 'RedAge' in df_features.columns:
            # Check first row
            expected_diff = df_features.loc[0, 'RedAge'] - df_features.loc[0, 'BlueAge']
            actual_diff = df_features.loc[0, 'Age_diff']
            
            assert np.isclose(expected_diff, actual_diff, equal_nan=True), \
                f"Age_diff calculation incorrect: expected {expected_diff}, got {actual_diff}"
    
    def test_feature_engineering_preserves_rows(self, sample_fight_data):
        """Test that feature engineering preserves number of rows."""
        original_rows = len(sample_fight_data)
        df_features = engineer_features(sample_fight_data.copy())
        
        assert len(df_features) == original_rows, \
            f"Feature engineering changed number of rows: {original_rows} -> {len(df_features)}"
    
    def test_momentum_features_range(self, sample_fight_data):
        """Test that momentum features are in valid ranges."""
        df_features = engineer_features(sample_fight_data.copy())
        
        if 'Red_win_streak' in df_features.columns:
            streaks = df_features['Red_win_streak'].dropna()
            assert (streaks >= 0).all(), "Win streaks should be >= 0"
        
        if 'Red_last3_win_rate' in df_features.columns:
            win_rates = df_features['Red_last3_win_rate'].dropna()
            assert (win_rates >= 0).all() and (win_rates <= 1).all(), \
                "Win rates should be between 0 and 1"


class TestDataSchema:
    """Test data schema validation."""
    
    def test_required_columns_present(self, sample_fight_data):
        """Test that all required columns are present."""
        required = ['RedFighter', 'BlueFighter', 'Winner']
        
        for col in required:
            assert col in sample_fight_data.columns, \
                f"Required column '{col}' is missing"
    
    def test_column_types_correct(self, sample_fight_data):
        """Test that column types are correct."""
        # Fighter names should be strings
        assert pd.api.types.is_string_dtype(sample_fight_data['RedFighter'])
        assert pd.api.types.is_string_dtype(sample_fight_data['BlueFighter'])
        
        # Winner should be string
        assert pd.api.types.is_string_dtype(sample_fight_data['Winner'])
    
    @pytest.mark.parametrize("numeric_col", [
        'RedAge', 'BlueAge', 'RedHeight', 'BlueHeight',
        'RedReach', 'BlueReach', 'RedWeight', 'BlueWeight'
    ])
    def test_numeric_columns_exist_and_numeric(self, sample_fight_data, numeric_col):
        """Test that numeric columns exist and are numeric."""
        if numeric_col in sample_fight_data.columns:
            assert pd.api.types.is_numeric_dtype(sample_fight_data[numeric_col]), \
                f"Column {numeric_col} should be numeric"


class TestDataAfterPreprocessing:
    """Test data validation after preprocessing."""
    
    def test_target_column_binary(self, sample_fight_data):
        """Test that target column is binary after preprocessing."""
        df = _create_target(sample_fight_data.copy())
        
        assert 'target' in df.columns, "Target column missing"
        assert df['target'].dtype in [np.int64, np.int32, np.int8], \
            "Target column should be integer type"
        assert set(df['target'].unique()).issubset({0, 1}), \
            "Target column should only contain 0 and 1"
    
    def test_no_infinite_values(self, sample_fight_data):
        """Test that there are no infinite values in numeric columns."""
        df = engineer_features(sample_fight_data.copy())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].dtype in [np.float64, np.float32]:
                assert np.isfinite(df[col].fillna(0)).all(), \
                    f"Column {col} contains infinite values"

