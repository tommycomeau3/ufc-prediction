"""
Integration tests for the prediction pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scripts.preprocess import clean_ufc_data, scale_features
from scripts.features import engineer_features
from scripts.matchup import build_matchup_row, predict_matchup, load_stats_cache
from scripts.model import train_model


class TestPredictionPipeline:
    """Test the complete prediction pipeline."""
    
    def test_end_to_end_prediction_pipeline(self, sample_fight_data, temp_model_dir, temp_data_dir):
        """Test complete pipeline from data to prediction."""
        # Save sample data
        data_path = temp_data_dir / "ufc_data.csv"
        sample_fight_data.to_csv(data_path, index=False)
        
        # Clean and engineer features
        df_clean = clean_ufc_data(data_path)
        df_features = engineer_features(df_clean)
        
        # Prepare features for training
        X, y = scale_features(df_features)
        
        # Train a simple model
        model = train_model(
            X, y,
            model_type="logreg",
            save_path=str(temp_model_dir / "test_model.pkl")
        )
        
        # Verify model was saved
        assert (temp_model_dir / "test_model.pkl").exists()
        
        # Verify model can make predictions
        predictions = model.predict(X[:1])
        probabilities = model.predict_proba(X[:1])
        
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
        assert probabilities.shape == (1, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_feature_engineering_pipeline_consistency(self, sample_fight_data):
        """Test that feature engineering produces consistent results."""
        # Engineer features twice
        df1 = engineer_features(sample_fight_data.copy())
        df2 = engineer_features(sample_fight_data.copy())
        
        # Should produce same features
        assert set(df1.columns) == set(df2.columns)
        
        # Differential features should be consistent
        if 'Age_diff' in df1.columns:
            assert np.allclose(
                df1['Age_diff'].fillna(0),
                df2['Age_diff'].fillna(0),
                equal_nan=True
            )


class TestMatchupPrediction:
    """Test matchup prediction functionality."""
    
    def test_build_matchup_row(self, sample_fighter_stats):
        """Test building a matchup row from fighter stats."""
        result = build_matchup_row('Fighter A', 'Fighter X', sample_fighter_stats)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.loc[0, 'RedFighter'] == 'Fighter A'
        assert result.loc[0, 'BlueFighter'] == 'Fighter X'
        assert 'Date' in result.columns
    
    def test_build_matchup_row_case_insensitive(self, sample_fighter_stats):
        """Test that fighter lookup is case-insensitive."""
        result = build_matchup_row('fighter a', 'FIGHTER X', sample_fighter_stats)
        
        assert result.loc[0, 'RedFighter'] == 'Fighter A'
        assert result.loc[0, 'BlueFighter'] == 'Fighter X'
    
    def test_build_matchup_row_fighter_not_found(self, sample_fighter_stats):
        """Test error handling when fighter is not found."""
        with pytest.raises(KeyError, match="not found"):
            build_matchup_row('Unknown Fighter', 'Fighter X', sample_fighter_stats)
    
    def test_build_matchup_row_populates_stats(self, sample_fighter_stats):
        """Test that fighter stats are properly populated."""
        result = build_matchup_row('Fighter A', 'Fighter X', sample_fighter_stats)
        
        # Check that Red stats are populated
        assert 'RedAge' in result.columns
        assert 'RedHeight' in result.columns
        assert not pd.isna(result.loc[0, 'RedAge'])
        
        # Check that Blue stats are populated
        assert 'BlueAge' in result.columns
        assert 'BlueHeight' in result.columns
        assert not pd.isna(result.loc[0, 'BlueAge'])


class TestModelTrainingIntegration:
    """Test model training integration."""
    
    def test_train_model_logistic_regression(self, sample_fight_data):
        """Test training logistic regression model."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_model(X, y, model_type="logreg")
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_model_gradient_boosting(self, sample_fight_data):
        """Test training gradient boosting model."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_model(X, y, model_type="gbdt")
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_model_invalid_type(self, sample_fight_data):
        """Test error handling for invalid model type."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(X, y, model_type="invalid")
    
    def test_train_model_saves_to_path(self, sample_fight_data, temp_model_dir):
        """Test that model is saved to specified path."""
        from scripts.preprocess import _create_target
        
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model_path = temp_model_dir / "custom_model.pkl"
        model = train_model(X, y, model_type="logreg", save_path=str(model_path))
        
        assert model_path.exists()
        
        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None


class TestStatsCache:
    """Test fighter statistics cache functionality."""
    
    def test_load_stats_cache_creates_file(self, sample_csv_file, temp_data_dir, monkeypatch):
        """Test that stats cache is created when it doesn't exist."""
        from scripts.matchup import STATS_CACHE
        
        # Mock the cache path to use temp directory
        cache_path = temp_data_dir / "fighter_stats.csv"
        
        # This test would require mocking the module-level constant
        # For now, we'll test the cache building logic
        from scripts.matchup import _build_stats_cache
        
        stats_df = _build_stats_cache(sample_csv_file)
        
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0
        assert 'fighter' in stats_df.index.name or 'fighter' in stats_df.columns


class TestFeaturePipelineConsistency:
    """Test that feature pipeline is consistent between training and prediction."""
    
    def test_feature_columns_match(self, sample_fight_data):
        """Test that feature engineering produces expected columns."""
        df = engineer_features(sample_fight_data.copy())
        
        # Should have original columns (unless drop_original=True)
        assert 'RedFighter' in df.columns
        assert 'BlueFighter' in df.columns
        
        # Should have differential features
        assert 'Age_diff' in df.columns or 'age_diff' in df.columns
        
        # Should have momentum features
        assert 'Red_win_streak' in df.columns
        assert 'win_streak_diff' in df.columns
    
    def test_preprocessing_pipeline_reproducibility(self, sample_fight_data):
        """Test that preprocessing produces reproducible results."""
        from scripts.preprocess import _create_target
        
        df1 = _create_target(sample_fight_data.copy())
        df2 = _create_target(sample_fight_data.copy())
        
        X1, y1 = scale_features(df1)
        X2, y2 = scale_features(df2)
        
        # Should produce same results
        np.testing.assert_array_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

