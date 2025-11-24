# Testing Guide

This document provides instructions for running and understanding the test suite for the UFC Fight Prediction System.

## Quick Start

1. **Install test dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run all tests:**
   ```bash
   pytest
   ```

3. **Run tests with coverage:**
   ```bash
   pytest --cov=scripts --cov=main --cov-report=html
   ```

4. **View coverage report:**
   ```bash
   open htmlcov/index.html  # macOS
   # or
   xdg-open htmlcov/index.html  # Linux
   ```

## Test Structure

The test suite is organized into the following categories:

### 1. Unit Tests

#### `tests/test_features.py`
Tests for feature engineering functions:
- Differential feature creation (Red vs Blue differences)
- Momentum features (win streaks, recent form)
- Physical mismatch features (height, reach, weight, age)
- Ratio features (striking, takedown, defense ratios)
- Context features (short notice flags, altitude)

**Key test classes:**
- `TestEngineerFeatures`: Main feature engineering function
- `TestPhysicalDiffs`: Physical difference calculations
- `TestMomentumFeatures`: Win streak and recent form
- `TestRecentFormFeatures`: Rolling averages and EWMA
- `TestRatioFeatures`: Performance ratio calculations
- `TestShortNoticeFlag`: Short notice fight detection
- `TestAltitudeFeatures`: Altitude-based features

#### `tests/test_preprocess.py`
Tests for data preprocessing functions:
- CSV reading with encoding fallback
- Date parsing and coercion
- Target variable creation
- High missing value column removal
- Feature scaling and encoding

**Key test classes:**
- `TestReadCSV`: CSV file reading
- `TestCoerceDates`: Date parsing
- `TestCreateTarget`: Binary target creation
- `TestDropHighMissing`: Column filtering
- `TestCleanUFCData`: Complete cleaning pipeline
- `TestScaleFeatures`: Feature scaling

### 2. Integration Tests

#### `tests/test_integration.py`
End-to-end pipeline tests:
- Complete prediction pipeline from data to prediction
- Feature engineering consistency
- Matchup prediction functionality
- Model training integration
- Stats cache operations

**Key test classes:**
- `TestPredictionPipeline`: Full pipeline tests
- `TestMatchupPrediction`: Fighter matchup functionality
- `TestModelTrainingIntegration`: Model training workflows
- `TestStatsCache`: Fighter statistics caching
- `TestFeaturePipelineConsistency`: Feature consistency

### 3. Model Validation Tests

#### `tests/test_model_validation.py`
Model performance and validation:
- Accuracy and ROC-AUC thresholds
- Probability validation (sum to 1, range [0,1])
- Model reproducibility
- Model comparison (LogReg vs GBDT)
- Evaluation metrics

**Key test classes:**
- `TestModelPerformance`: Performance metrics
- `TestModelConsistency`: Reproducibility and consistency
- `TestModelComparison`: Model type comparison
- `TestModelEvaluation`: Evaluation functions

### 4. Data Validation Tests

#### `tests/test_data_validation.py`
Data quality and validation:
- Required columns presence
- Data completeness checks
- Data consistency validation
- Feature engineering output validation
- Schema validation
- Post-processing validation

**Key test classes:**
- `TestDataQuality`: Basic data quality checks
- `TestDataCompleteness`: Missing value handling
- `TestDataConsistency`: Data consistency checks
- `TestFeatureEngineeringValidation`: Feature validation
- `TestDataSchema`: Schema validation
- `TestDataAfterPreprocessing`: Post-processing checks

## Running Specific Tests

### Run a specific test file
```bash
pytest tests/test_features.py
```

### Run a specific test class
```bash
pytest tests/test_features.py::TestEngineerFeatures
```

### Run a specific test function
```bash
pytest tests/test_features.py::TestEngineerFeatures::test_engineer_features_basic
```

### Run tests matching a pattern
```bash
pytest -k "momentum"  # Runs all tests with "momentum" in the name
```

### Run tests by marker
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m model         # Model validation tests
pytest -m data          # Data validation tests
```

## Test Fixtures

Shared test fixtures are defined in `tests/conftest.py`:

- **`sample_fight_data`**: Complete sample fight data
- **`sample_fight_data_with_missing`**: Data with missing values
- **`sample_fight_data_high_missing`**: Data with high missing rates
- **`sample_fighter_stats`**: Fighter statistics DataFrame
- **`trained_logistic_model`**: Pre-trained logistic regression model
- **`trained_gbdt_model`**: Pre-trained gradient boosting model
- **`temp_data_dir`**: Temporary directory for test data files
- **`temp_model_dir`**: Temporary directory for test model files

## Coverage Goals

- **Minimum coverage**: 70% (enforced in pytest.ini)
- **Target coverage**: 80%+
- **Critical paths**: 100% coverage

Current coverage can be viewed by running:
```bash
pytest --cov=scripts --cov=main --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

When adding new functionality, follow these guidelines:

1. **File naming**: Use `test_*.py` format
2. **Function naming**: Use `test_*` prefix
3. **Class naming**: Use `Test*` prefix for test classes
4. **Use fixtures**: Leverage shared fixtures from `conftest.py`
5. **Parameterize**: Use `@pytest.mark.parametrize` for multiple scenarios
6. **Documentation**: Include docstrings explaining what is tested
7. **Assertions**: Use descriptive assertion messages

### Example Test

```python
import pytest
from scripts.features import engineer_features

class TestNewFeature:
    """Test new feature engineering function."""
    
    def test_new_feature_creation(self, sample_fight_data):
        """Test that new feature is created correctly."""
        result = engineer_features(sample_fight_data.copy())
        
        assert 'new_feature' in result.columns
        assert result['new_feature'].notna().any()
    
    @pytest.mark.parametrize("input_value,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_new_feature_calculation(self, input_value, expected):
        """Test new feature calculation with different inputs."""
        # Test implementation
        assert input_value * 2 == expected
```

## Continuous Integration

The test suite is configured for CI/CD:

- **Coverage reporting**: Term, HTML, and XML formats
- **Minimum coverage**: 70% threshold enforced
- **Test discovery**: Automatic discovery of test files
- **Warning filters**: Deprecation warnings filtered

## Troubleshooting

### Import Errors
If you see import errors:
1. Ensure you're in the project root directory
2. Install all dependencies: `pip install -r requirements.txt`
3. Check that Python path includes project root

### Fixture Not Found
If fixtures aren't found:
1. Verify `conftest.py` exists in `tests/` directory
2. Check fixture names match exactly
3. Ensure pytest version is up to date: `pytest --version`

### Coverage Below Threshold
If coverage is below 70%:
1. Run with verbose output: `pytest -v --cov=scripts`
2. Check HTML report for uncovered lines
3. Add tests for uncovered code paths

### Tests Fail Due to Missing Data
Some tests may require actual data files:
1. Ensure `data/ufc-master.csv` exists (for integration tests)
2. Tests use fixtures for most scenarios, but some integration tests need real data
3. Mock data files when possible using fixtures

## Best Practices

1. **Isolation**: Each test should be independent
2. **Speed**: Keep tests fast (< 1 second each when possible)
3. **Clarity**: Test names should clearly describe what is tested
4. **Coverage**: Aim for high coverage of critical paths
5. **Maintenance**: Update tests when code changes
6. **Documentation**: Document complex test scenarios

## Next Steps

After running tests:
1. Review coverage report
2. Add tests for uncovered code
3. Fix any failing tests
4. Update tests when adding new features
5. Consider adding performance benchmarks

