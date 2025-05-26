# GridSource Installation Guide

## Quick Setup for Jupyter Notebooks

To use the GridSource package components in Jupyter notebooks with proper imports, you need to install the package in editable mode.

### 1. Install the Package

From the GridSource root directory, run:

```bash
# Install in editable mode (development install)
pip install -e .

# Or install with all dependencies for testing and AWS
pip install -e ".[testing,aws,snowflake]"
```

### 2. Verify Installation

```python
# Test imports in Python/Jupyter
from gridsource.research.data_helpers import APIExplorer
from gridsource.tests.unit.extraction_functions import extract_eia_electricity_data_test
from gridsource.sagemaker.train import LiquidityForecastingModel

print("✅ All imports working!")
```

### 3. Available Imports

After installation, you can use these imports in your notebooks:

```python
# Research and data exploration
from gridsource.research.data_helpers import (
    APIExplorer, 
    get_synchronized_ml_data, 
    quick_explore_apis
)

# Testing functions
from gridsource.tests.unit.extraction_functions import (
    extract_eia_electricity_data_test,
    transform_eia_data,
    create_ml_features_test
)

# ML training
from gridsource.sagemaker.train import LiquidityForecastingModel
```

## Why This Approach?

### Before (Path Manipulation):
```python
import sys
sys.path.append('..')  # Fragile, error-prone
from research.data_helpers import APIExplorer  # May not work
```

### After (Proper Package Install):
```python
from gridsource.research.data_helpers import APIExplorer  # Always works
```

## Benefits

1. **✅ Clean Imports**: No path manipulation needed
2. **✅ Consistent**: Works from any directory
3. **✅ IDE Support**: Better autocomplete and error checking
4. **✅ Production Ready**: Same approach used for real packages
5. **✅ Dependency Management**: Automatic dependency installation

## Troubleshooting

### Import Errors
If you get import errors:

```bash
# Reinstall in editable mode
pip uninstall gridsource
pip install -e .
```

### Missing Dependencies
```bash
# Install specific dependency groups
pip install -e ".[testing]"      # For testing
pip install -e ".[aws]"          # For AWS/Airflow
pip install -e ".[snowflake]"    # For Snowflake
```

### Jupyter Kernel Issues
```bash
# Restart Jupyter kernel after installation
# Kernel -> Restart & Clear Output
```

## Development Workflow

1. **Make code changes** in any GridSource module
2. **No reinstall needed** (editable install automatically picks up changes)
3. **Restart Jupyter kernel** if you change function signatures
4. **Import and use** your updated functions

This approach follows Python packaging best practices and makes your code much more maintainable!