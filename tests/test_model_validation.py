"""
Model validation tests.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from scripts.model import train_model, train_and_evaluate_model
from scripts.preprocess import _create_target, scale_features
from scripts.features import engineer_features


class TestModelPerformance:
    """Test model performance metrics."""
    
    def test_model_accuracy_threshold(self, sample_fight_data):
        """Test that model achieves minimum accuracy threshold."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = train_model(X_train, y_train, model_type="logreg")
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # For a binary classifier with balanced data, expect > 50% accuracy
        assert accuracy > 0.5, f"Model accuracy {accuracy:.2%} is too low"
    
    def test_model_roc_auc_threshold(self, sample_fight_data):
        """Test that model achieves minimum ROC-AUC threshold."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = train_model(X_train, y_train, model_type="logreg")
        
        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # ROC-AUC should be > 0.5 (better than random)
        assert roc_auc > 0.5, f"Model ROC-AUC {roc_auc:.2%} is too low"
    
    def test_model_probabilities_sum_to_one(self, sample_fight_data):
        """Test that model probabilities sum to 1."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_model(X, y, model_type="logreg")
        probabilities = model.predict_proba(X)
        
        # Probabilities should sum to 1 for each sample
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_probabilities_range(self, sample_fight_data):
        """Test that model probabilities are in valid range [0, 1]."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_model(X, y, model_type="logreg")
        probabilities = model.predict_proba(X)
        
        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)


class TestModelConsistency:
    """Test model consistency and reproducibility."""
    
    def test_model_reproducibility(self, sample_fight_data):
        """Test that model training is reproducible with same random seed."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Train two models with same seed
        model1 = train_model(X, y, model_type="logreg", random_state=42)
        model2 = train_model(X, y, model_type="logreg", random_state=42)
        
        # Predictions should be identical
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_model_predictions_binary(self, sample_fight_data):
        """Test that model predictions are binary (0 or 1)."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_model(X, y, model_type="logreg")
        predictions = model.predict(X)
        
        # All predictions should be 0 or 1
        assert np.all(np.isin(predictions, [0, 1]))
    
    def test_model_handles_new_data(self, sample_fight_data):
        """Test that model can handle new data with same feature structure."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Split into train and "new" data
        X_train, X_new, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train on training data
        model = train_model(X_train, y_train, model_type="logreg")
        
        # Should be able to predict on new data
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        
        assert len(predictions) == len(X_new)
        assert probabilities.shape == (len(X_new), 2)


class TestModelComparison:
    """Test comparison between different model types."""
    
    def test_logreg_vs_gbdt_same_data(self, sample_fight_data):
        """Test that both model types can be trained on same data."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Train both models
        logreg_model = train_model(X, y, model_type="logreg")
        gbdt_model = train_model(X, y, model_type="gbdt")
        
        # Both should make predictions
        logreg_pred = logreg_model.predict(X)
        gbdt_pred = gbdt_model.predict(X)
        
        assert len(logreg_pred) == len(gbdt_pred)
        assert len(logreg_pred) == len(X)
    
    @pytest.mark.parametrize("model_type", ["logreg", "gbdt"])
    def test_model_types_train_successfully(self, sample_fight_data, model_type):
        """Test that different model types train successfully."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_model(X, y, model_type=model_type)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')


class TestModelEvaluation:
    """Test model evaluation functions."""
    
    def test_train_and_evaluate_model_output(self, sample_fight_data):
        """Test that train_and_evaluate_model produces valid output."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        model = train_and_evaluate_model(X, y)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_evaluation_metrics(self, sample_fight_data):
        """Test that model evaluation produces valid metrics."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = train_model(X_train, y_train, model_type="logreg")
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Metrics should be valid
        assert 0 <= accuracy <= 1
        assert 0 <= roc_auc <= 1
    
    def test_classification_report_format(self, sample_fight_data):
        """Test that classification report is properly formatted."""
        df = _create_target(sample_fight_data.copy())
        df = engineer_features(df)
        
        X, y = scale_features(df)
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = train_model(X_train, y_train, model_type="logreg")
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Report should have expected structure
        assert '0' in report or '1' in report
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report

