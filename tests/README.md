# Unit Tests

This directory contains unit tests for the Azure Machine Learning AutoML Training Framework.

## Test Structure

- `test_feature_engineering.py` - Tests for the feature engineering module
  - Tests for FeatureEngineer class and all transformation methods
  - Tests for data validation functions
  - 24 test cases covering binning, scaling, encoding, datetime features, interactions, polynomial features, and custom transformations

- `test_validate_config.py` - Tests for the configuration validation module
  - Tests for ConfigValidator class
  - Tests for all validation rules (experiment, data, feature engineering, compute, automl, output)
  - 28 test cases covering various configuration scenarios and edge cases

## Running Tests

### Prerequisites

Install the required dependencies:

```bash
pip install pytest pytest-cov
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run all tests with coverage report
pytest tests/ --cov=feature_engineering --cov=validate_config --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Run only feature engineering tests
pytest tests/test_feature_engineering.py -v

# Run only configuration validation tests
pytest tests/test_validate_config.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_feature_engineering.py::TestFeatureEngineer -v

# Run a specific test method
pytest tests/test_feature_engineering.py::TestFeatureEngineer::test_binning_basic -v
```

## Test Coverage

Current test coverage:
- **feature_engineering.py**: 93% coverage
- **validate_config.py**: 78% coverage
- **Overall**: 85% coverage

The tests cover:
- ✅ All feature engineering transformations
- ✅ Data validation rules
- ✅ Configuration validation for all sections
- ✅ Edge cases and error handling
- ✅ Invalid inputs and missing data scenarios

## Test Organization

### Feature Engineering Tests (`test_feature_engineering.py`)

**TestFeatureEngineer class**:
- Initialization and configuration
- Binning transformations (with valid and invalid inputs)
- Datetime feature extraction
- Scaling (standard, minmax, robust)
- Encoding (one-hot, label)
- Interaction features
- Polynomial features
- Custom transformations (including error cases)
- Multiple transformations applied together

**TestValidateData class**:
- Missing values detection and removal
- Duplicate rows detection and removal
- Validation thresholds
- Combined validation checks

### Configuration Validation Tests (`test_validate_config.py`)

**TestConfigValidator class**:
- File existence and YAML syntax validation
- Required sections validation
- Experiment configuration validation
- Data configuration validation (paths, file types, required fields)
- Feature engineering validation:
  - Binning (bins/labels matching)
  - Datetime features (valid extract types)
  - Interactions (pair structure)
  - Custom transformations (required fields)
- Compute configuration validation
- AutoML configuration validation (task types, metrics, training parameters)
- Output configuration validation (log levels)

## Writing New Tests

When adding new functionality, follow these guidelines:

1. **Test File Naming**: Use `test_<module_name>.py` format
2. **Test Class Naming**: Use `Test<ClassName>` format
3. **Test Method Naming**: Use `test_<descriptive_name>` format with clear descriptions
4. **Test Structure**:
   ```python
   def test_feature_name(self):
       """Clear description of what is being tested"""
       # Arrange: Set up test data and configuration
       config = {...}
       
       # Act: Execute the function/method being tested
       result = function_under_test(config)
       
       # Assert: Verify the results
       assert result == expected_value
   ```

5. **Use pytest fixtures** for common test data or setup
6. **Test both success and failure cases**
7. **Use descriptive assertion messages** when needed
8. **Keep tests independent** - each test should run in isolation

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```bash
# Run tests with JUnit XML output for CI
pytest tests/ --junitxml=test-results.xml

# Run tests with coverage in CI
pytest tests/ --cov --cov-report=xml --cov-report=html
```

## Troubleshooting

### Common Issues

**Import errors**: Make sure you're running pytest from the repository root directory:
```bash
cd /path/to/upmc-mlstudio-automl-jobs
pytest tests/
```

**Missing dependencies**: Install all required packages:
```bash
pip install -r requirements.txt
```

**Test failures**: Check the detailed error output with `-v` flag:
```bash
pytest tests/ -v
```

## Future Improvements

Areas for additional test coverage:
- Integration tests for the full training pipeline (requires Azure credentials/mocking)
- Tests for deploy.py module
- Tests for inference.py module
- Tests for train.py module (requires Azure ML SDK mocking)
- Performance tests for large datasets
- End-to-end tests with sample data files
