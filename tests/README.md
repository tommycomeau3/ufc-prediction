# Test Suite Documentation

This directory contains comprehensive tests for the UFC Fight Prediction System.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures
├── test_features.py         # Unit tests for feature engineering
├── test_preprocess.py       # Unit tests for data preprocessing
├── test_integration.py      # Integration tests for prediction pipeline
├── test_model_validation.py # Model validation and performance tests
└── test_data_validation.py  # Data quality and validation tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=scripts --cov=main --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_features.py
```

### Run specific test class
```bash
pytest tests/test_features.py::TestEngineerFeatures
```

### Run specific test function
```bash
pytest tests/test_features.py::TestEngineerFeatures::test_engineer_features_basic
```

### Run tests by marker
```bash
pytest -m unit          # Run only unit tests
pytest -m integration   # Run only integration tests
pytest -m model         # Run only model validation tests
pytest -m data           # Run only data validation tests
```

### Run tests in verbose mode
```bash
pytest -v
```

### Run tests and show print statements
```bash
pytest -s
```

## Test Categories

### Unit Tests
- **test_features.py**: Tests for feature engineering functions
  - Differential feature creation
  - Momentum features
  - Recent form features
  - Ratio features
  - Physical differences
  - Short notice flags
  - Altitude features

- **test_preprocess.py**: Tests for data preprocessing functions
  - CSV reading with encoding fallback
  - Date coercion
  - Target variable creation
  - High missing value column dropping
  - Feature scaling and encoding

### Integration Tests
- **test_integration.py**: End-to-end pipeline tests
  - Complete prediction pipeline
  - Feature engineering consistency
  - Matchup prediction
  - Model training integration
  - Stats cache functionality

### Model Validation Tests
- **test_model_validation.py**: Model performance and validation
  - Accuracy thresholds
  - ROC-AUC thresholds
  - Probability validation
  - Model reproducibility
  - Model comparison
  - Evaluation metrics

### Data Validation Tests
- **test_data_validation.py**: Data quality checks
  - Required columns
  - Data completeness
  - Data consistency
  - Feature engineering validation
  - Schema validation
  - Post-processing validation

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_fight_data`: Sample fight data with all required columns
- `sample_fight_data_with_missing`: Sample data with missing values
- `sample_fight_data_high_missing`: Sample data with high missing values
- `sample_fighter_stats`: Sample fighter statistics
- `mock_model_pipeline`: Mock model pipeline
- `trained_logistic_model`: Trained logistic regression model
- `trained_gbdt_model`: Trained gradient boosting model
- `temp_data_dir`: Temporary directory for test data
- `temp_model_dir`: Temporary directory for test models
- `sample_csv_file`: Temporary CSV file
- `sample_model_file`: Temporary model file

## Coverage Goals

The test suite aims for:
- **Minimum coverage**: 70% (configured in pytest.ini)
- **Target coverage**: 80%+
- **Critical paths**: 100% coverage

## Continuous Integration

Tests are designed to run in CI/CD pipelines. The pytest configuration includes:
- Coverage reporting (term, HTML, XML)
- Minimum coverage threshold enforcement
- Warning filters
- Test discovery patterns

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `test_*` for functions
2. Use fixtures from `conftest.py` when possible
3. Use parameterized tests for testing multiple scenarios
4. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
5. Include docstrings explaining what is being tested
6. Use descriptive test names that explain the expected behavior

## Example Test

```python
def test_feature_engineering_creates_diffs(sample_fight_data):
    """Test that feature engineering creates differential features."""
    result = engineer_features(sample_fight_data.copy())
    
    assert 'Age_diff' in result.columns
    assert 'Height_diff' in result.columns
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
- The project root is in Python path
- All dependencies are installed: `pip install -r requirements.txt`
- You're running tests from the project root

### Fixture Errors
If fixtures are not found:
- Check that `conftest.py` is in the `tests/` directory
- Ensure fixture names match exactly
- Verify pytest version: `pytest --version`

### Coverage Issues
If coverage is below threshold:
- Run with verbose output: `pytest -v --cov=scripts`
- Check HTML report: `htmlcov/index.html`
- Review which lines are not covered

