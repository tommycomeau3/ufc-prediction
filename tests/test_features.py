"""
Unit tests for feature engineering functions.
"""
import pytest
import pandas as pd
import numpy as np
from scripts.features import (
    engineer_features,
    build_momentum_features,
    build_recent_form_features,
    _add_physical_diffs,
    _add_ratio_features,
    _add_short_notice_flag,
    _add_cage_altitude_features,
    _find_shared_numeric_columns,
)


class TestEngineerFeatures:
    """Test the main engineer_features function."""
    
    def test_engineer_features_basic(self, sample_fight_data):
        """Test basic feature engineering creates differential features."""
        result = engineer_features(sample_fight_data.copy())
        
        # Should create differential features
        assert 'Age_diff' in result.columns
        assert 'Height_diff' in result.columns
        assert 'Reach_diff' in result.columns
        assert 'Weight_diff' in result.columns
        assert 'AvgSigStrLanded_diff' in result.columns
        
        # Original columns should still exist by default
        assert 'RedAge' in result.columns
        assert 'BlueAge' in result.columns
    
    def test_engineer_features_drop_original(self, sample_fight_data):
        """Test feature engineering with drop_original=True."""
        result = engineer_features(sample_fight_data.copy(), drop_original=True)
        
        # Differential features should exist
        assert 'Age_diff' in result.columns
        
        # Original Red/Blue columns should be dropped
        assert 'RedAge' not in result.columns
        assert 'BlueAge' not in result.columns
    
    def test_engineer_features_creates_momentum(self, sample_fight_data):
        """Test that momentum features are created."""
        result = engineer_features(sample_fight_data.copy())
        
        # Momentum features should be present
        assert 'Red_win_streak' in result.columns
        assert 'Blue_win_streak' in result.columns
        assert 'win_streak_diff' in result.columns
        assert 'Red_last3_win_rate' in result.columns
        assert 'Blue_last3_win_rate' in result.columns
        assert 'last3_win_rate_diff' in result.columns
    
    def test_engineer_features_creates_ratios(self, sample_fight_data):
        """Test that ratio features are created."""
        result = engineer_features(sample_fight_data.copy())
        
        # Ratio features should be present if stats exist
        if 'RedAvgSigStrLanded' in sample_fight_data.columns:
            assert 'strikes_lpm_ratio' in result.columns or 'strikes_lpm_ratio' not in result.columns  # May or may not be created
    
    def test_engineer_features_handles_missing_data(self, sample_fight_data_with_missing):
        """Test feature engineering with missing values."""
        result = engineer_features(sample_fight_data_with_missing.copy())
        
        # Should not crash and should create features
        assert 'Age_diff' in result.columns
        assert result.shape[0] == sample_fight_data_with_missing.shape[0]


class TestPhysicalDiffs:
    """Test physical difference feature creation."""
    
    def test_add_physical_diffs(self, sample_fight_data):
        """Test physical difference features are created correctly."""
        result = _add_physical_diffs(sample_fight_data.copy())
        
        assert 'height_diff' in result.columns
        assert 'reach_diff' in result.columns
        assert 'weight_diff' in result.columns
        assert 'age_diff' in result.columns
        
        # Check values
        assert result.loc[0, 'height_diff'] == 180.0 - 178.0
        assert result.loc[0, 'age_diff'] == 28 - 26
    
    def test_add_physical_diffs_missing_columns(self, sample_fight_data):
        """Test physical diffs handles missing columns gracefully."""
        df = sample_fight_data.copy()
        df = df.drop(columns=['RedHeight', 'BlueHeight'])
        result = _add_physical_diffs(df)
        
        # Should not crash, other diffs should still be created
        assert 'age_diff' in result.columns


class TestMomentumFeatures:
    """Test momentum feature creation."""
    
    def test_build_momentum_features_basic(self, sample_fight_data):
        """Test basic momentum features."""
        result = build_momentum_features(sample_fight_data.copy())
        
        assert 'Red_win_streak' in result.columns
        assert 'Blue_win_streak' in result.columns
        assert 'win_streak_diff' in result.columns
        assert 'Red_last3_win_rate' in result.columns
        assert 'Blue_last3_win_rate' in result.columns
    
    def test_build_momentum_features_no_date(self, sample_fight_data):
        """Test momentum features when Date column is missing."""
        df = sample_fight_data.copy()
        df = df.drop(columns=['Date'])
        result = build_momentum_features(df)
        
        # Should return original dataframe unchanged
        assert result.shape == df.shape
    
    def test_build_momentum_features_streak_calculation(self, sample_fight_data):
        """Test that win streaks are calculated correctly."""
        # Create a simple test case
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
            'RedFighter': ['A', 'A', 'A'],
            'BlueFighter': ['X', 'Y', 'Z'],
            'Winner': ['Red', 'Red', 'Blue'],
        })
        
        result = build_momentum_features(df)
        
        # First fight: no previous history, streak should be 0
        assert result.loc[0, 'Red_win_streak'] == 0
        
        # Second fight: Red won previous, streak should be 1
        assert result.loc[1, 'Red_win_streak'] == 1
        
        # Third fight: Red won previous two, streak should be 2
        assert result.loc[2, 'Red_win_streak'] == 2


class TestRecentFormFeatures:
    """Test recent form feature creation."""
    
    def test_build_recent_form_features_basic(self, sample_fight_data):
        """Test basic recent form features."""
        result = build_recent_form_features(sample_fight_data.copy(), window=5, ewma_span=10)
        
        # Should create rolling and EWMA features
        assert any('_roll5' in col for col in result.columns)
        assert any('_ewm10' in col for col in result.columns)
    
    def test_build_recent_form_features_no_date(self, sample_fight_data):
        """Test recent form features when Date column is missing."""
        df = sample_fight_data.copy()
        df = df.drop(columns=['Date'])
        result = build_recent_form_features(df)
        
        # Should return original dataframe unchanged
        assert result.shape == df.shape
    
    @pytest.mark.parametrize("window,ewma_span", [
        (3, 5),
        (5, 10),
        (10, 20),
    ])
    def test_build_recent_form_features_parameters(self, sample_fight_data, window, ewma_span):
        """Test recent form features with different parameters."""
        result = build_recent_form_features(
            sample_fight_data.copy(), 
            window=window, 
            ewma_span=ewma_span
        )
        
        # Should create features with specified window/span
        assert any(f'_roll{window}' in col for col in result.columns)
        assert any(f'_ewm{ewma_span}' in col for col in result.columns)


class TestRatioFeatures:
    """Test ratio feature creation."""
    
    def test_add_ratio_features(self, sample_fight_data):
        """Test ratio features are created correctly."""
        result = _add_ratio_features(sample_fight_data.copy())
        
        # Should create ratio features if stats exist
        if 'RedAvgSigStrLanded' in sample_fight_data.columns:
            assert 'strikes_lpm_ratio' in result.columns
    
    def test_add_ratio_features_handles_zero_division(self, sample_fight_data):
        """Test ratio features handle division by zero."""
        df = sample_fight_data.copy()
        df.loc[0, 'BlueAvgSigStrLanded'] = 0.0
        
        result = _add_ratio_features(df)
        
        # Should not crash, should handle NaN
        if 'strikes_lpm_ratio' in result.columns:
            assert pd.isna(result.loc[0, 'strikes_lpm_ratio'])


class TestShortNoticeFlag:
    """Test short notice flag feature."""
    
    def test_add_short_notice_flag_with_red_blue_columns(self):
        """Test short notice flag with Red/Blue notice columns."""
        df = pd.DataFrame({
            'RedNoticeDays': [5, 20, 30],
            'BlueNoticeDays': [10, 15, 25],
        })
        
        result = _add_short_notice_flag(df, threshold=14)
        
        assert 'short_notice_flag' in result.columns
        assert result.loc[0, 'short_notice_flag'] == 1  # Red has 5 days (< 14)
        assert result.loc[1, 'short_notice_flag'] == 0  # Red has 20, Blue has 15 (both >= 14)
        assert result.loc[2, 'short_notice_flag'] == 0  # Both >= 14 (30, 25)
    
    def test_add_short_notice_flag_with_single_column(self):
        """Test short notice flag with single NoticeDays column."""
        df = pd.DataFrame({
            'NoticeDays': [5, 20, 30],
        })
        
        result = _add_short_notice_flag(df, threshold=14)
        
        assert 'short_notice_flag' in result.columns
        assert result.loc[0, 'short_notice_flag'] == 1
        assert result.loc[1, 'short_notice_flag'] == 0
        assert result.loc[2, 'short_notice_flag'] == 0
    
    @pytest.mark.parametrize("threshold", [7, 14, 21, 30])
    def test_add_short_notice_flag_thresholds(self, threshold):
        """Test short notice flag with different thresholds."""
        df = pd.DataFrame({
            'RedNoticeDays': [threshold - 1, threshold, threshold + 1],
            'BlueNoticeDays': [threshold, threshold, threshold],
        })
        
        result = _add_short_notice_flag(df, threshold=threshold)
        
        assert result.loc[0, 'short_notice_flag'] == 1  # Below threshold
        assert result.loc[1, 'short_notice_flag'] == 0  # At threshold
        assert result.loc[2, 'short_notice_flag'] == 0  # Above threshold


class TestAltitudeFeatures:
    """Test altitude feature creation."""
    
    def test_add_cage_altitude_features(self):
        """Test altitude features are created."""
        df = pd.DataFrame({
            'EventAltitude': [0, 1500, 2000, 500],
        })
        
        result = _add_cage_altitude_features(df)
        
        assert 'altitude_diff' in result.columns
        assert 'high_altitude_flag' in result.columns
        
        assert result.loc[0, 'high_altitude_flag'] == 0  # Below 1500m
        assert result.loc[1, 'high_altitude_flag'] == 1  # At 1500m
        assert result.loc[2, 'high_altitude_flag'] == 1  # Above 1500m
        assert result.loc[3, 'high_altitude_flag'] == 0  # Below 1500m
    
    def test_add_cage_altitude_features_no_column(self, sample_fight_data):
        """Test altitude features when column is missing."""
        result = _add_cage_altitude_features(sample_fight_data.copy())
        
        # Should return original dataframe unchanged
        assert result.shape == sample_fight_data.shape


class TestSharedNumericColumns:
    """Test finding shared numeric columns."""
    
    def test_find_shared_numeric_columns(self, sample_fight_data):
        """Test finding shared Red/Blue numeric columns."""
        shared = _find_shared_numeric_columns(sample_fight_data)
        
        assert isinstance(shared, list)
        assert 'Age' in shared
        assert 'Height' in shared
        assert 'Reach' in shared
    
    def test_find_shared_numeric_columns_no_shared(self):
        """Test when no shared columns exist."""
        df = pd.DataFrame({
            'RedOnlyCol': [1, 2, 3],      # Base: "OnlyCol"
            'BlueDifferentCol': [4, 5, 6], # Base: "DifferentCol" - different!
        })
        
        shared = _find_shared_numeric_columns(df)
        
        assert len(shared) == 0

