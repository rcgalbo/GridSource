"""
Unit tests for data extraction functions

Tests each data extraction component independently with mocked API responses
"""

import pytest
import pandas as pd
import json
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.conftest import MockResponse


class TestDataExtraction:
    """Test data extraction functions from the Airflow DAG"""

    @pytest.mark.unit
    def test_eia_data_extraction_success(self, sample_eia_data, mock_requests, mock_s3_setup):
        """Test successful EIA data extraction"""
        s3_client, bucket_name = mock_s3_setup
        
        # Mock successful API response
        mock_requests.return_value = MockResponse(sample_eia_data)
        
        # Import and test the extraction function
        from tests.unit.extraction_functions import extract_eia_electricity_data_test
        
        result_path = extract_eia_electricity_data_test(
            api_key='test_key',
            s3_client=s3_client,
            bucket_name=bucket_name
        )
        
        # Verify S3 upload
        assert result_path.startswith(f's3://{bucket_name}/raw/eia/')
        
        # Verify data was uploaded to S3
        objects = s3_client.list_objects(Bucket=bucket_name, Prefix='raw/eia/')
        assert 'Contents' in objects
        assert len(objects['Contents']) == 1

    @pytest.mark.unit
    def test_eia_data_transformation(self, sample_eia_data):
        """Test EIA data transformation logic"""
        from tests.unit.extraction_functions import transform_eia_data
        
        df = transform_eia_data(sample_eia_data)
        
        # Check DataFrame structure
        expected_columns = ['date', 'fuel_type', 'generation_mwh', 'data_source']
        assert list(df.columns) == expected_columns
        
        # Check data types
        assert df['date'].dtype == 'datetime64[ns]'
        assert df['generation_mwh'].dtype == 'float64'
        assert df['data_source'].iloc[0] == 'EIA'
        
        # Check data content
        assert len(df) == 4  # 4 records in sample data
        assert df['fuel_type'].unique().tolist() == ['NG', 'SUN', 'WND']

    @pytest.mark.unit
    def test_weather_data_extraction_success(self, sample_weather_data, mock_requests, mock_s3_setup):
        """Test successful NOAA weather data extraction"""
        s3_client, bucket_name = mock_s3_setup
        
        # Mock successful API response
        mock_requests.return_value = MockResponse(sample_weather_data)
        
        from tests.unit.extraction_functions import extract_weather_data_test
        
        result_path = extract_weather_data_test(
            s3_client=s3_client,
            bucket_name=bucket_name
        )
        
        # Verify S3 upload
        assert result_path.startswith(f's3://{bucket_name}/raw/weather/')
        
        # Verify data was uploaded
        objects = s3_client.list_objects(Bucket=bucket_name, Prefix='raw/weather/')
        assert 'Contents' in objects

    @pytest.mark.unit
    def test_weather_data_transformation(self, sample_weather_data):
        """Test weather data transformation logic"""
        from tests.unit.extraction_functions import transform_weather_data
        
        df = transform_weather_data(sample_weather_data)
        
        # Check DataFrame structure
        expected_columns = ['date', 'temperature_f', 'wind_speed', 'forecast', 'data_source']
        assert list(df.columns) == expected_columns
        
        # Check data content
        assert len(df) == 3
        assert df['data_source'].iloc[0] == 'NOAA'
        assert df['temperature_f'].dtype == 'int64'

    @pytest.mark.unit
    def test_fred_data_extraction_success(self, sample_fred_data, mock_requests, mock_s3_setup):
        """Test successful FRED economic data extraction"""
        s3_client, bucket_name = mock_s3_setup
        
        # Mock successful API response for each indicator
        mock_requests.return_value = MockResponse(sample_fred_data)
        
        from tests.unit.extraction_functions import extract_fred_data_test
        
        result_path = extract_fred_data_test(
            api_key='test_key',
            s3_client=s3_client,
            bucket_name=bucket_name
        )
        
        # Verify S3 upload
        assert result_path.startswith(f's3://{bucket_name}/raw/economic/')

    @pytest.mark.unit
    def test_fred_data_transformation(self, sample_fred_data):
        """Test FRED data transformation logic"""
        from tests.unit.extraction_functions import transform_fred_data
        
        df = transform_fred_data(sample_fred_data, 'industrial_production_index')
        
        # Check DataFrame structure
        expected_columns = ['date', 'indicator', 'value', 'data_source']
        assert list(df.columns) == expected_columns
        
        # Check data content
        assert len(df) == 4
        assert df['data_source'].iloc[0] == 'FRED'
        assert df['indicator'].iloc[0] == 'industrial_production_index'
        assert df['value'].dtype == 'float64'

    @pytest.mark.unit
    def test_energy_price_generation(self, mock_s3_setup):
        """Test energy price data generation"""
        s3_client, bucket_name = mock_s3_setup
        
        from tests.unit.extraction_functions import generate_energy_prices_test
        
        result_path = generate_energy_prices_test(
            s3_client=s3_client,
            bucket_name=bucket_name,
            days=5
        )
        
        # Verify S3 upload
        assert result_path.startswith(f's3://{bucket_name}/raw/prices/')

    @pytest.mark.unit
    def test_api_error_handling(self, mock_requests, mock_s3_setup):
        """Test API error handling"""
        s3_client, bucket_name = mock_s3_setup
        
        # Mock API error
        mock_requests.return_value = MockResponse({}, status_code=500)
        
        from tests.unit.extraction_functions import extract_eia_electricity_data_test
        
        with pytest.raises(Exception):
            extract_eia_electricity_data_test(
                api_key='test_key',
                s3_client=s3_client,
                bucket_name=bucket_name
            )

    @pytest.mark.unit
    def test_invalid_api_key(self, mock_requests, mock_s3_setup):
        """Test invalid API key handling"""
        s3_client, bucket_name = mock_s3_setup
        
        # Mock API authentication error
        mock_requests.return_value = MockResponse(
            {'error': 'Invalid API key'}, 
            status_code=401
        )
        
        from tests.unit.extraction_functions import extract_eia_electricity_data_test
        
        with pytest.raises(Exception):
            extract_eia_electricity_data_test(
                api_key='invalid_key',
                s3_client=s3_client,
                bucket_name=bucket_name
            )

    @pytest.mark.unit
    def test_data_validation(self, sample_eia_data):
        """Test data validation logic"""
        from tests.unit.extraction_functions import validate_eia_data
        
        # Test valid data
        is_valid, error_msg = validate_eia_data(sample_eia_data)
        assert is_valid is True
        assert error_msg is None
        
        # Test invalid data
        invalid_data = {'response': {'data': []}}
        is_valid, error_msg = validate_eia_data(invalid_data)
        assert is_valid is False
        assert 'No data found' in error_msg

    @pytest.mark.unit
    def test_s3_upload_error_handling(self, sample_eia_data, mock_requests):
        """Test S3 upload error handling"""
        # Mock successful API response
        mock_requests.return_value = MockResponse(sample_eia_data)
        
        # Mock S3 client that raises an error
        mock_s3_client = Mock()
        mock_s3_client.put_object.side_effect = Exception("S3 upload failed")
        
        from tests.unit.extraction_functions import extract_eia_electricity_data_test
        
        with pytest.raises(Exception, match="S3 upload failed"):
            extract_eia_electricity_data_test(
                api_key='test_key',
                s3_client=mock_s3_client,
                bucket_name='test-bucket'
            )

    @pytest.mark.unit
    def test_date_range_generation(self):
        """Test date range generation for API calls"""
        from tests.unit.extraction_functions import generate_date_range
        
        start_date, end_date = generate_date_range(days=7)
        
        # Check date format
        assert len(start_date) == 10  # YYYY-MM-DD format
        assert len(end_date) == 10
        assert start_date < end_date
        
        # Check date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        diff = (end_dt - start_dt).days
        assert diff == 7

    @pytest.mark.unit
    def test_data_cleaning(self):
        """Test data cleaning functions"""
        from tests.unit.extraction_functions import clean_numeric_data
        
        # Test with valid data
        clean_value = clean_numeric_data('123.45')
        assert clean_value == 123.45
        
        # Test with invalid data
        clean_value = clean_numeric_data('.')
        assert clean_value is None
        
        # Test with missing data
        clean_value = clean_numeric_data(None)
        assert clean_value is None