"""
Unit tests for SageMaker ML training components

Tests the machine learning model training logic independently
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestLiquidityForecastingModel:
    """Test the LiquidityForecastingModel class"""

    @pytest.mark.unit
    def test_model_initialization(self):
        """Test model initialization with different types"""
        from sagemaker.train import LiquidityForecastingModel
        
        # Test linear regression
        model = LiquidityForecastingModel(model_type='linear_regression')
        assert model.model_type == 'linear_regression'
        assert model.model is not None
        
        # Test ridge regression
        model = LiquidityForecastingModel(model_type='ridge')
        assert model.model_type == 'ridge'
        
        # Test random forest
        model = LiquidityForecastingModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        
        # Test invalid model type
        with pytest.raises(ValueError, match="Unknown model type"):
            LiquidityForecastingModel(model_type='invalid_model')

    @pytest.mark.unit
    def test_data_loading_and_preparation(self, sample_training_data_csv):
        """Test loading and preparing training data"""
        from sagemaker.train import LiquidityForecastingModel
        
        model = LiquidityForecastingModel()
        X, y = model.load_and_prepare_data(sample_training_data_csv)
        
        # Check data shapes
        assert X.shape[0] == 5  # 5 samples
        assert X.shape[1] == 5  # 5 features
        assert len(y) == 5
        
        # Check feature names
        expected_features = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        assert model.feature_names == expected_features
        
        # Check data types
        assert X.dtypes.apply(lambda x: x.kind in 'biufc').all()  # Numeric types

    @pytest.mark.unit
    def test_data_loading_missing_columns(self, temp_directory):
        """Test error handling for missing columns"""
        from sagemaker.train import LiquidityForecastingModel
        
        # Create CSV with missing columns
        incomplete_data = pd.DataFrame({
            'total_generation_mwh': [75000, 76000],
            'avg_temperature_f': [65, 67]
            # Missing required columns
        })
        
        csv_path = os.path.join(temp_directory, 'incomplete_data.csv')
        incomplete_data.to_csv(csv_path, index=False)
        
        model = LiquidityForecastingModel()
        
        with pytest.raises(ValueError, match="Missing required feature columns"):
            model.load_and_prepare_data(csv_path)

    @pytest.mark.unit
    def test_model_training(self, sample_ml_features):
        """Test model training process"""
        from sagemaker.train import LiquidityForecastingModel
        
        model = LiquidityForecastingModel(model_type='linear_regression')
        
        # Prepare data
        feature_cols = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        X = sample_ml_features[feature_cols]
        y = sample_ml_features['liquidity_need_millions']
        
        # Train model
        model.feature_names = feature_cols
        model.train(X, y, test_size=0.3)
        
        # Check that model was trained
        assert model.model is not None
        assert hasattr(model.model, 'predict')
        
        # Check training metrics
        assert 'train' in model.training_metrics
        assert 'validation' in model.training_metrics
        assert 'mae' in model.training_metrics['train']
        assert 'r2' in model.training_metrics['train']

    @pytest.mark.unit
    def test_model_prediction(self, sample_ml_features):
        """Test model prediction functionality"""
        from sagemaker.train import LiquidityForecastingModel
        
        model = LiquidityForecastingModel(model_type='linear_regression')
        
        # Prepare and train model
        feature_cols = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        X = sample_ml_features[feature_cols]
        y = sample_ml_features['liquidity_need_millions']
        
        model.feature_names = feature_cols
        model.train(X, y)
        
        # Make predictions
        predictions = model.predict(X.head(3))
        
        # Check prediction output
        assert len(predictions) == 3
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    @pytest.mark.unit
    def test_model_saving_and_loading(self, sample_ml_features, temp_directory):
        """Test model saving and loading"""
        from sagemaker.train import LiquidityForecastingModel
        
        # Train model
        model = LiquidityForecastingModel(model_type='ridge')
        
        feature_cols = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        X = sample_ml_features[feature_cols]
        y = sample_ml_features['liquidity_need_millions']
        
        model.feature_names = feature_cols
        model.train(X, y)
        
        # Save model
        model_dir = os.path.join(temp_directory, 'model')
        model.save_model(model_dir)
        
        # Check that files were created
        assert os.path.exists(os.path.join(model_dir, 'model.pkl'))
        assert os.path.exists(os.path.join(model_dir, 'scaler.pkl'))
        assert os.path.exists(os.path.join(model_dir, 'feature_names.json'))
        assert os.path.exists(os.path.join(model_dir, 'metadata.json'))
        
        # Test loading
        loaded_model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        loaded_scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        assert loaded_model is not None
        assert loaded_scaler is not None

    @pytest.mark.unit
    def test_metric_calculation(self, sample_ml_features):
        """Test model performance metric calculation"""
        from sagemaker.train import LiquidityForecastingModel
        
        model = LiquidityForecastingModel()
        
        # Create some test predictions
        y_true = np.array([100, 150, 200, 250, 300])
        y_pred = np.array([105, 145, 195, 255, 295])
        
        metrics = model._calculate_metrics(y_true, y_pred, "Test")
        
        # Check that all expected metrics are present
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # Check metric values are reasonable
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert 0 <= metrics['r2'] <= 1
        assert metrics['mape'] > 0

    @pytest.mark.unit
    def test_feature_importance_extraction(self, sample_ml_features):
        """Test feature importance extraction for tree-based models"""
        from sagemaker.train import LiquidityForecastingModel
        
        model = LiquidityForecastingModel(model_type='random_forest')
        
        feature_cols = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        X = sample_ml_features[feature_cols]
        y = sample_ml_features['liquidity_need_millions']
        
        model.feature_names = feature_cols
        model.train(X, y)
        
        # Check that feature importances exist for random forest
        assert hasattr(model.model, 'feature_importances_')
        assert len(model.model.feature_importances_) == len(feature_cols)

    @pytest.mark.unit
    def test_data_cleaning_missing_values(self, temp_directory):
        """Test handling of missing values in training data"""
        from sagemaker.train import LiquidityForecastingModel
        
        # Create data with missing values
        data_with_nulls = {
            'total_generation_mwh': [75000, None, 77000, 78000, 79000],
            'avg_temperature_f': [65, 67, None, 68, 72],
            'oil_price_usd': [70.5, 71.0, 69.8, None, 70.9],
            'industrial_production_index': [101.5, 101.6, 101.7, 101.8, 101.9],
            'avg_electricity_price': [45.0, 46.2, 44.8, 47.1, 45.5],
            'liquidity_need_millions': [150.0, 152.5, 155.0, 157.5, 160.0]
        }
        
        df = pd.DataFrame(data_with_nulls)
        csv_path = os.path.join(temp_directory, 'data_with_nulls.csv')
        df.to_csv(csv_path, index=False)
        
        model = LiquidityForecastingModel()
        X, y = model.load_and_prepare_data(csv_path)
        
        # Check that missing values were handled
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    @pytest.mark.unit
    def test_prediction_without_training(self, sample_ml_features):
        """Test error when trying to predict without training"""
        from sagemaker.train import LiquidityForecastingModel
        
        model = LiquidityForecastingModel()
        
        feature_cols = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        X = sample_ml_features[feature_cols].head(3)
        
        with pytest.raises(ValueError, match="Model not trained yet"):
            model.predict(X)


class TestTrainingScript:
    """Test the main training script functionality"""

    @pytest.mark.unit
    def test_argument_parsing(self):
        """Test command line argument parsing"""
        from sagemaker.train import parse_arguments
        
        # Test with default arguments
        with patch('sys.argv', ['train.py']):
            args = parse_arguments()
            assert args.model_dir == '/opt/ml/model'
            assert args.train == '/opt/ml/input/data/training'
            assert args.model_type == 'linear_regression'

    @pytest.mark.unit  
    def test_main_training_function(self, sample_training_data_csv, temp_directory):
        """Test the main training function"""
        from sagemaker.train import main
        
        # Create mock arguments
        with patch('sagemaker.train.parse_arguments') as mock_args:
            mock_args.return_value = Mock(
                model_dir=temp_directory,
                train=os.path.dirname(sample_training_data_csv),
                model_type='linear_regression',
                test_size=0.2,
                random_state=42
            )
            
            # Run main function
            main()
            
            # Check that model artifacts were created
            assert os.path.exists(os.path.join(temp_directory, 'model.pkl'))
            assert os.path.exists(os.path.join(temp_directory, 'metadata.json'))

    @pytest.mark.unit
    def test_training_with_different_model_types(self, sample_training_data_csv, temp_directory):
        """Test training with different model types"""
        from sagemaker.train import LiquidityForecastingModel
        
        model_types = ['linear_regression', 'ridge', 'random_forest']
        
        for model_type in model_types:
            model = LiquidityForecastingModel(model_type=model_type)
            X, y = model.load_and_prepare_data(sample_training_data_csv)
            
            # Train model
            model.train(X, y)
            
            # Check that training completed successfully
            assert model.training_metrics is not None
            assert model.model is not None