"""
Pytest configuration and shared fixtures for GridSource testing

This module provides common test fixtures and configuration for testing
the GridSource Bank energy banking pipeline components.
"""

import pytest
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import boto3
from moto import mock_s3, mock_sagemaker
import tempfile
import shutil


@pytest.fixture
def sample_eia_data():
    """Sample EIA electricity generation data for testing"""
    return {
        'response': {
            'data': [
                {
                    'period': '2025-05-23',
                    'fueltype': 'NG',
                    'value': 25000.5,
                    'respondent': 'CAL'
                },
                {
                    'period': '2025-05-23',
                    'fueltype': 'SUN',
                    'value': 15000.0,
                    'respondent': 'CAL'
                },
                {
                    'period': '2025-05-22',
                    'fueltype': 'NG',
                    'value': 26000.0,
                    'respondent': 'CAL'
                },
                {
                    'period': '2025-05-22',
                    'fueltype': 'WND',
                    'value': 8000.0,
                    'respondent': 'CAL'
                }
            ]
        }
    }


@pytest.fixture
def sample_weather_data():
    """Sample NOAA weather data for testing"""
    return {
        'properties': {
            'periods': [
                {
                    'name': 'Today',
                    'startTime': '2025-05-23T06:00:00-07:00',
                    'temperature': 72,
                    'windSpeed': '10 mph',
                    'shortForecast': 'Partly Cloudy'
                },
                {
                    'name': 'Tonight',
                    'startTime': '2025-05-23T18:00:00-07:00',
                    'temperature': 58,
                    'windSpeed': '5 mph',
                    'shortForecast': 'Clear'
                },
                {
                    'name': 'Tomorrow',
                    'startTime': '2025-05-24T06:00:00-07:00',
                    'temperature': 75,
                    'windSpeed': '12 mph',
                    'shortForecast': 'Sunny'
                }
            ]
        }
    }


@pytest.fixture
def sample_fred_data():
    """Sample FRED economic data for testing"""
    return {
        'observations': [
            {
                'date': '2025-05-01',
                'value': '102.5'
            },
            {
                'date': '2025-04-01',
                'value': '101.8'
            },
            {
                'date': '2025-03-01',
                'value': '101.2'
            },
            {
                'date': '2025-02-01',
                'value': '100.9'
            }
        ]
    }


@pytest.fixture
def sample_ml_features():
    """Sample ML features dataframe for testing"""
    dates = pd.date_range(start='2025-05-01', end='2025-05-20', freq='D')
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'total_generation_mwh': 75000 + i * 1000,
            'avg_temperature_f': 65 + (i % 15),
            'oil_price_usd': 70.5 + (i % 10),
            'industrial_production_index': 101.5 + (i * 0.1),
            'avg_electricity_price': 45.0 + (i % 8),
            'liquidity_need_millions': 150.0 + (i * 2.5)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing"""
    return {
        'eia_api_key': 'test_eia_key_123',
        'fred_api_key': 'test_fred_key_456'
    }


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_s3_setup():
    """Set up mock S3 environment for testing"""
    with mock_s3():
        # Create mock S3 client
        s3_client = boto3.client('s3', region_name='us-west-2')
        
        # Create test bucket
        bucket_name = 'test-grid-source-data'
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
        )
        
        yield s3_client, bucket_name


@pytest.fixture
def mock_sagemaker_setup():
    """Set up mock SageMaker environment for testing"""
    with mock_sagemaker():
        sagemaker_client = boto3.client('sagemaker', region_name='us-west-2')
        yield sagemaker_client


@pytest.fixture
def sample_training_data_csv(temp_directory):
    """Create sample training data CSV file for testing"""
    data = {
        'total_generation_mwh': [75000, 76000, 77000, 78000, 79000],
        'avg_temperature_f': [65, 67, 70, 68, 72],
        'oil_price_usd': [70.5, 71.0, 69.8, 72.1, 70.9],
        'industrial_production_index': [101.5, 101.6, 101.7, 101.8, 101.9],
        'avg_electricity_price': [45.0, 46.2, 44.8, 47.1, 45.5],
        'liquidity_need_millions': [150.0, 152.5, 155.0, 157.5, 160.0]
    }
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_directory, 'training_data.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def mock_snowflake_connection():
    """Mock Snowflake connection for testing"""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Mock cursor methods
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    
    # Mock connection methods
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.close.return_value = None
    
    return mock_conn, mock_cursor


@pytest.fixture
def sample_airflow_context():
    """Sample Airflow context for testing tasks"""
    return {
        'ds': '2025-05-23',
        'ds_nodash': '20250523',
        'ts': '2025-05-23T10:00:00+00:00',
        'ts_nodash': '20250523T100000',
        'dag': Mock(),
        'task': Mock(),
        'task_instance': Mock()
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
    yield
    # Cleanup
    if 'ENVIRONMENT' in os.environ:
        del os.environ['ENVIRONMENT']


class MockResponse:
    """Mock HTTP response for API testing"""
    
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data) if isinstance(json_data, dict) else str(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code} Error")


@pytest.fixture
def mock_requests():
    """Mock requests module for API testing"""
    with patch('requests.get') as mock_get:
        yield mock_get


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring external API"
    )