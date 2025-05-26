# GridSource Testing & Research Framework

This document explains the comprehensive testing and research framework for the GridSource Bank energy banking pipeline.

## 🧪 Testing Framework Overview

The testing framework is designed to make pipeline development and debugging significantly easier by allowing you to test each component independently.

### Why This Testing Approach?

**Traditional Pipeline Problem**: 
```
API → Airflow → S3 → Snowflake → SageMaker → Snowflake → Power BI
```
If something fails, it's hard to know where the issue is!

**Our Solution**:
```
✅ Test Each Component Independently
✅ Use Mock Data for Isolation  
✅ Validate Data Transformations
✅ Debug Issues Quickly
```

## 📁 Testing Structure

```
tests/
├── conftest.py                    # Shared test fixtures and configuration
├── requirements.txt               # Testing dependencies
├── run_tests.py                  # Test runner script
├── unit/
│   ├── test_data_extraction.py   # Test API data extraction
│   ├── test_ml_training.py       # Test ML model training
│   └── extraction_functions.py   # Testable extraction functions
├── integration/
│   └── test_pipeline_integration.py  # Test component interactions
└── fixtures/                     # Test data files
```

## 🚀 Quick Start - Running Tests

### 1. Install Testing Dependencies

```bash
cd tests
pip install -r requirements.txt
```

### 2. Run All Tests

```bash
# From project root
python tests/run_tests.py --type all --verbose
```

### 3. Run Specific Test Types

```bash
# Unit tests only
python tests/run_tests.py --type unit

# Integration tests only  
python tests/run_tests.py --type integration

# With coverage report
python tests/run_tests.py --type coverage
```

### 4. Run Specific Tests

```bash
# Test specific function
python tests/run_tests.py --test tests/unit/test_data_extraction.py::TestDataExtraction::test_eia_data_transformation

# Test specific file
python tests/run_tests.py --test tests/unit/test_ml_training.py
```

### 5. Check Test Environment

```bash
python tests/run_tests.py --check
```

## 🔍 What Each Test Does

### Unit Tests (`tests/unit/`)

#### Data Extraction Tests (`test_data_extraction.py`)
- **EIA API Testing**: Validates electricity generation data transformation
- **Weather API Testing**: Tests NOAA weather data processing
- **FRED API Testing**: Validates economic indicator data handling
- **Error Handling**: Tests API failures and invalid responses
- **Data Validation**: Checks data quality and completeness

#### ML Training Tests (`test_ml_training.py`)
- **Model Initialization**: Tests different model types (Linear, Ridge, Random Forest)
- **Data Loading**: Validates training data preparation
- **Training Process**: Tests model training with sample data
- **Prediction**: Validates model prediction functionality
- **Model Persistence**: Tests saving and loading trained models

### Integration Tests (`tests/integration/`)

#### Pipeline Integration (`test_pipeline_integration.py`)
- **End-to-End Flow**: Tests complete data pipeline
- **Component Interaction**: Validates data flow between components
- **S3 Integration**: Tests S3 upload/download functionality
- **Data Consistency**: Validates data schemas across transformations
- **Error Propagation**: Tests how errors flow through the pipeline

## 🔧 Testable Functions (`tests/unit/extraction_functions.py`)

We've created standalone, testable versions of all pipeline functions:

### Data Extraction Functions
```python
# Test EIA data extraction
extract_eia_electricity_data_test(api_key, s3_client, bucket_name)

# Test weather data extraction  
extract_weather_data_test(s3_client, bucket_name)

# Test economic data extraction
extract_fred_data_test(api_key, s3_client, bucket_name)

# Test energy price generation
generate_energy_prices_test(s3_client, bucket_name)
```

### Data Transformation Functions
```python
# Transform API responses to DataFrames
transform_eia_data(api_response)
transform_weather_data(api_response)
transform_fred_data(api_response, indicator_name)

# Create ML features from all sources
create_ml_features_test(eia_df, weather_df, economic_df, price_df)
```

## 📊 Research Framework

The research directory provides tools for exploring APIs and understanding data patterns.

### Research Structure

```
research/
├── data_helpers.py               # API exploration helpers
└── notebooks/
    ├── 01_EIA_API_Exploration.ipynb
    ├── 02_Weather_API_Exploration.ipynb
    └── 03_Pipeline_Testing_Demo.ipynb
```

### Using Research Tools

```python
from research.data_helpers import APIExplorer, quick_explore_apis

# Initialize explorer with API keys
explorer = APIExplorer({
    'eia_api_key': 'your_key_here',
    'fred_api_key': 'your_key_here'
})

# Quick exploration of all APIs
results = quick_explore_apis()

# Individual API exploration
eia_data = explorer.explore_eia_api(days=14)
weather_data = explorer.explore_weather_api()
fred_data = explorer.explore_fred_api()

# Analyze patterns
explorer.compare_data_sources(eia_data, weather_data, fred_data)
```

## 🎯 Testing Examples

### Example 1: Test EIA Data Transformation

```python
# Sample API response
sample_response = {
    'response': {
        'data': [
            {
                'period': '2025-05-23',
                'fueltype': 'NG',
                'value': '25000.5',
                'respondent': 'CAL'
            }
        ]
    }
}

# Test transformation
from tests.unit.extraction_functions import transform_eia_data
df = transform_eia_data(sample_response)

# Validate results
assert len(df) == 1
assert df['fuel_type'].iloc[0] == 'NG'
assert df['generation_mwh'].iloc[0] == 25000.5
```

### Example 2: Test ML Model Training

```python
from sagemaker.train import LiquidityForecastingModel
import pandas as pd

# Create sample training data
data = {
    'total_generation_mwh': [75000, 76000, 77000],
    'avg_temperature_f': [65, 67, 70],
    'oil_price_usd': [70.5, 71.0, 69.8],
    'industrial_production_index': [101.5, 101.6, 101.7],
    'avg_electricity_price': [45.0, 46.2, 44.8],
    'liquidity_need_millions': [150.0, 152.5, 155.0]
}

# Train model
model = LiquidityForecastingModel()
X = pd.DataFrame({k: v for k, v in data.items() if k != 'liquidity_need_millions'})
y = pd.Series(data['liquidity_need_millions'])

model.feature_names = list(X.columns)
model.train(X, y)

# Test predictions
predictions = model.predict(X)
assert len(predictions) == len(y)
```

### Example 3: Test End-to-End Pipeline

```python
from tests.unit.extraction_functions import create_ml_features_test

# Sample data from each source
eia_df = pd.DataFrame({
    'date': ['2025-05-23', '2025-05-23'],
    'fuel_type': ['NG', 'SUN'],
    'generation_mwh': [25000, 15000]
})

weather_df = pd.DataFrame({
    'date': ['2025-05-23'],
    'temperature_f': [72]
})

# Create ML features
features = create_ml_features_test(eia_df, weather_df, economic_df, price_df)

# Validate feature engineering
assert 'total_generation_mwh' in features.columns
assert 'liquidity_need_millions' in features.columns
assert len(features) > 0
```

## 🎓 Learning Benefits

### For Pipeline Development
- **🔍 Isolated Debugging**: Test each component separately
- **🚀 Faster Development**: No need to run entire pipeline for small changes
- **📊 Data Quality**: Catch data issues early in development
- **🔄 Reproducible**: Same results across different environments

### For Learning Data Engineering
- **Clear Code Structure**: Functions are easy to understand and modify
- **Minimal Abstraction**: Direct, readable code without complex frameworks
- **Real API Integration**: Work with actual external APIs
- **Best Practices**: Demonstrates modern testing approaches

## 🔧 Customizing Tests

### Adding New Tests

1. **Create test file** in appropriate directory (`unit/` or `integration/`)
2. **Use fixtures** from `conftest.py` for common test data
3. **Follow naming convention**: `test_*.py` for files, `test_*` for functions
4. **Use descriptive assertions** with clear error messages

### Example New Test

```python
# tests/unit/test_new_feature.py
import pytest
from your_module import new_function

class TestNewFeature:
    
    @pytest.mark.unit
    def test_new_function_success(self, sample_data):
        """Test new function with valid input"""
        result = new_function(sample_data)
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.unit  
    def test_new_function_error_handling(self):
        """Test new function with invalid input"""
        with pytest.raises(ValueError):
            new_function(invalid_data)
```

## 📈 Test Coverage Goals

Current test coverage targets:

- **✅ Data Extraction**: 100% (All APIs covered)
- **✅ Data Transformation**: 100% (All transform functions)
- **✅ ML Training**: 95% (Core functionality)
- **✅ Feature Engineering**: 90% (Key transformations)
- **⚠️ Airflow Integration**: Manual testing (requires Airflow setup)
- **⚠️ Snowflake Integration**: Manual testing (requires Snowflake access)

## 🚨 Troubleshooting Tests

### Common Issues

#### Tests Not Found
```bash
# Make sure you're in the project root
cd /path/to/GridSource
python tests/run_tests.py --check
```

#### Import Errors
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/GridSource"
```

#### API Key Issues
```bash
# Tests should work without API keys (use mock data)
# Check that tests use sample data when API keys aren't available
```

#### Mock Data Issues
```bash
# Verify mock data fixtures in conftest.py
python -c "from tests.conftest import sample_eia_data; print('✅ Fixtures working')"
```

## 🎯 Next Steps

1. **Run the tests** to understand current pipeline status
2. **Explore notebooks** to understand API data structures
3. **Modify test data** to match your specific use cases
4. **Add new tests** as you extend the pipeline
5. **Set up CI/CD** for automated testing in production

---

**Happy Testing!** 🧪✨

This testing framework makes pipeline development much more manageable and helps you understand exactly how each component works. Start with the research notebooks to explore the data, then use the tests to validate your pipeline components as you build them out.