"""
Integration tests for the complete pipeline

Tests the interaction between different components of the pipeline
"""

import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, Mock
import boto3
from moto import mock_s3, mock_sagemaker

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestPipelineIntegration:
    """Test integration between pipeline components"""

    @pytest.mark.integration
    def test_end_to_end_data_flow(self, sample_eia_data, sample_weather_data, 
                                  sample_fred_data, mock_s3_setup):
        """Test complete data flow from extraction to feature creation"""
        s3_client, bucket_name = mock_s3_setup
        
        # Import extraction functions
        from tests.unit.extraction_functions import (
            transform_eia_data, transform_weather_data, 
            transform_fred_data, create_ml_features_test
        )
        
        # Transform all data sources
        eia_df = transform_eia_data(sample_eia_data)
        weather_df = transform_weather_data(sample_weather_data)
        economic_df = transform_fred_data(sample_fred_data, 'industrial_production_index')
        
        # Create sample price data
        price_df = pd.DataFrame({
            'date': ['2025-05-22', '2025-05-23'],
            'price_per_mwh': [45.0, 47.0]
        })
        
        # Create ML features
        features_df = create_ml_features_test(eia_df, weather_df, economic_df, price_df)
        
        # Verify feature creation
        assert len(features_df) > 0
        assert 'total_generation_mwh' in features_df.columns
        assert 'liquidity_need_millions' in features_df.columns

    @pytest.mark.integration
    def test_s3_to_ml_training_flow(self, sample_ml_features, mock_s3_setup, temp_directory):
        """Test data flow from S3 to ML training"""
        s3_client, bucket_name = mock_s3_setup
        
        # Upload training data to S3
        training_data_key = 'ml/training_data.csv'
        csv_content = sample_ml_features.to_csv(index=False)
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=training_data_key,
            Body=csv_content
        )
        
        # Download and train model
        from sagemaker.train import LiquidityForecastingModel
        
        # Download from S3
        local_file = os.path.join(temp_directory, 'training_data.csv')
        s3_client.download_file(bucket_name, training_data_key, local_file)
        
        # Train model
        model = LiquidityForecastingModel()
        X, y = model.load_and_prepare_data(local_file)
        model.train(X, y)
        
        # Verify training
        assert model.training_metrics is not None
        assert 'validation' in model.training_metrics

    @pytest.mark.integration
    def test_airflow_task_dependencies(self):
        """Test that Airflow task dependencies are correctly defined"""
        # Import the DAG
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'airflow', 'dags'))
        
        # This would require Airflow to be installed and configured
        # For now, we'll test the DAG structure
        
        # Import DAG
        try:
            from grid_source_pipeline import grid_source_dag
            
            # Check that DAG exists
            assert grid_source_dag is not None
            assert grid_source_dag.dag_id == 'grid_source_automated_pipeline'
            
            # Check task count (approximate)
            tasks = grid_source_dag.task_dict
            assert len(tasks) > 10  # Should have multiple tasks
            
        except ImportError:
            pytest.skip("Airflow not available for integration testing")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_model_performance_consistency(self, sample_ml_features):
        """Test that model performance is consistent across multiple runs"""
        from sagemaker.train import LiquidityForecastingModel
        
        feature_cols = [
            'total_generation_mwh',
            'avg_temperature_f',
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        X = sample_ml_features[feature_cols]
        y = sample_ml_features['liquidity_need_millions']
        
        # Train multiple models with same random seed
        r2_scores = []
        for _ in range(3):
            model = LiquidityForecastingModel(
                model_type='linear_regression',
                random_state=42
            )
            model.feature_names = feature_cols
            model.train(X, y, test_size=0.3)
            r2_scores.append(model.training_metrics['validation']['r2'])
        
        # Check consistency (should be identical with same random seed)
        assert all(abs(score - r2_scores[0]) < 0.001 for score in r2_scores)

    @pytest.mark.integration
    def test_data_validation_pipeline(self, mock_s3_setup):
        """Test data validation across the pipeline"""
        s3_client, bucket_name = mock_s3_setup
        
        from tests.unit.extraction_functions import validate_eia_data
        
        # Test valid data
        valid_data = {
            'response': {
                'data': [
                    {'period': '2025-05-23', 'fueltype': 'NG', 'value': '25000'},
                    {'period': '2025-05-22', 'fueltype': 'SUN', 'value': '15000'}
                ]
            }
        }
        
        is_valid, error = validate_eia_data(valid_data)
        assert is_valid is True
        assert error is None
        
        # Test invalid data
        invalid_data = {'response': {'data': []}}
        is_valid, error = validate_eia_data(invalid_data)
        assert is_valid is False
        assert error is not None

    @pytest.mark.integration
    def test_feature_engineering_consistency(self):
        """Test that feature engineering produces consistent results"""
        from tests.unit.extraction_functions import create_ml_features_test
        
        # Create consistent test data
        eia_df = pd.DataFrame({
            'date': ['2025-05-23', '2025-05-23', '2025-05-22'],
            'fuel_type': ['NG', 'SUN', 'NG'],
            'generation_mwh': [25000, 15000, 26000]
        })
        
        weather_df = pd.DataFrame({
            'date': ['2025-05-23', '2025-05-22'],
            'temperature_f': [72, 68]
        })
        
        economic_df = pd.DataFrame({
            'date': ['2025-05-23', '2025-05-22'],
            'indicator': ['oil_price', 'oil_price'],
            'value': [70.5, 71.0]
        })
        
        price_df = pd.DataFrame({
            'date': ['2025-05-23', '2025-05-22'],
            'price_per_mwh': [45.0, 47.0]
        })
        
        # Run feature engineering multiple times
        features_1 = create_ml_features_test(eia_df, weather_df, economic_df, price_df)
        features_2 = create_ml_features_test(eia_df, weather_df, economic_df, price_df)
        
        # Check deterministic columns (non-random ones)
        pd.testing.assert_series_equal(
            features_1['total_generation_mwh'], 
            features_2['total_generation_mwh']
        )

    @pytest.mark.integration
    def test_error_propagation(self, mock_s3_setup):
        """Test that errors propagate correctly through the pipeline"""
        s3_client, bucket_name = mock_s3_setup
        
        from tests.unit.extraction_functions import extract_eia_electricity_data_test
        
        # Test with invalid S3 client
        mock_s3_client = Mock()
        mock_s3_client.put_object.side_effect = Exception("S3 connection failed")
        
        with pytest.raises(Exception, match="S3 connection failed"):
            extract_eia_electricity_data_test(
                api_key='test_key',
                s3_client=mock_s3_client,
                bucket_name=bucket_name
            )

    @pytest.mark.integration
    def test_data_schema_consistency(self, sample_eia_data, sample_weather_data, sample_fred_data):
        """Test that data schemas are consistent across transformations"""
        from tests.unit.extraction_functions import (
            transform_eia_data, transform_weather_data, transform_fred_data
        )
        
        # Transform data
        eia_df = transform_eia_data(sample_eia_data)
        weather_df = transform_weather_data(sample_weather_data)
        economic_df = transform_fred_data(sample_fred_data, 'test_indicator')
        
        # Check expected columns
        assert list(eia_df.columns) == ['date', 'fuel_type', 'generation_mwh', 'data_source']
        assert list(weather_df.columns) == ['date', 'temperature_f', 'wind_speed', 'forecast', 'data_source']
        assert list(economic_df.columns) == ['date', 'indicator', 'value', 'data_source']
        
        # Check data types
        assert eia_df['date'].dtype == 'datetime64[ns]'
        assert eia_df['generation_mwh'].dtype == 'float64'
        assert weather_df['temperature_f'].dtype == 'int64'
        assert economic_df['value'].dtype == 'float64'

    @pytest.mark.integration
    def test_scalability_simulation(self, mock_s3_setup):
        """Test pipeline behavior with larger datasets"""
        s3_client, bucket_name = mock_s3_setup
        
        # Create large synthetic dataset
        import numpy as np
        
        dates = pd.date_range(start='2023-01-01', end='2025-05-23', freq='D')
        large_eia_data = []
        
        for date in dates[:100]:  # Test with 100 days to keep test fast
            for fuel in ['NG', 'SUN', 'WAT', 'WND']:
                large_eia_data.append({
                    'period': date.strftime('%Y-%m-%d'),
                    'fueltype': fuel,
                    'value': str(np.random.uniform(10000, 50000)),
                    'respondent': 'CAL'
                })
        
        large_sample_data = {'response': {'data': large_eia_data}}
        
        from tests.unit.extraction_functions import transform_eia_data
        
        # Test transformation performance
        df = transform_eia_data(large_sample_data)
        
        # Verify results
        assert len(df) == 400  # 100 days * 4 fuel types
        assert df['generation_mwh'].notna().all()
        assert len(df['fuel_type'].unique()) == 4