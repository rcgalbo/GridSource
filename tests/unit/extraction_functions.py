"""
Testable extraction functions separated from Airflow DAG

These functions mirror the logic in the main DAG but are structured
for independent testing without Airflow dependencies.
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def generate_date_range(days: int = 30) -> Tuple[str, str]:
    """Generate start and end dates for API calls"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def clean_numeric_data(value: str) -> Optional[float]:
    """Clean and convert numeric data from APIs"""
    if value is None or value == '.' or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def validate_eia_data(data: Dict) -> Tuple[bool, Optional[str]]:
    """Validate EIA API response data"""
    if 'response' not in data:
        return False, "Missing 'response' key in API data"
    
    if 'data' not in data['response']:
        return False, "Missing 'data' key in response"
    
    if not data['response']['data']:
        return False, "No data found in API response"
    
    return True, None


def transform_eia_data(api_data: Dict) -> pd.DataFrame:
    """Transform EIA API data to DataFrame"""
    data_records = api_data['response']['data']
    
    df = pd.DataFrame(data_records)
    df['date'] = pd.to_datetime(df['period'])
    df['generation_mwh'] = pd.to_numeric(df['value'], errors='coerce')
    df['fuel_type'] = df['fueltype']
    df['data_source'] = 'EIA'
    
    # Select and clean columns
    final_df = df[['date', 'fuel_type', 'generation_mwh', 'data_source']].dropna()
    return final_df


def extract_eia_electricity_data_test(api_key: str, s3_client, bucket_name: str, days: int = 30) -> str:
    """
    Test version of EIA electricity data extraction
    
    Args:
        api_key: EIA API key
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        days: Number of days of data to fetch
        
    Returns:
        S3 path to uploaded data
    """
    print(f"Extracting EIA data for last {days} days...")
    
    # API configuration
    base_url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
    start_date, end_date = generate_date_range(days)
    
    # API parameters
    params = {
        'api_key': api_key,
        'frequency': 'daily',
        'data[0]': 'value',
        'facets[respondent][]': 'CAL',
        'facets[fueltype][]': ['NG', 'SUN', 'WAT', 'WND'],
        'start': start_date,
        'end': end_date,
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'length': 1000
    }
    
    # Make API request
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Validate data
    is_valid, error_msg = validate_eia_data(data)
    if not is_valid:
        raise ValueError(f"Invalid EIA data: {error_msg}")
    
    print(f"Retrieved {len(data['response']['data'])} records from EIA")
    
    # Transform data
    df = transform_eia_data(data)
    
    # Upload to S3
    file_key = f'raw/eia/eia_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_buffer = df.to_csv(index=False)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_buffer
    )
    
    print(f"Uploaded EIA data to s3://{bucket_name}/{file_key}")
    return f's3://{bucket_name}/{file_key}'


def transform_weather_data(api_data: Dict) -> pd.DataFrame:
    """Transform NOAA weather API data to DataFrame"""
    weather_records = []
    
    for period in api_data['properties']['periods']:
        # Extract temperature
        temp = period.get('temperature', 0)
        
        # Parse wind speed (handle formats like "10 mph")
        wind_speed_raw = period.get('windSpeed', '0 mph')
        wind_speed = int(''.join(filter(str.isdigit, wind_speed_raw)) or '0')
        
        weather_records.append({
            'date': period['startTime'][:10],  # Extract date part only
            'temperature_f': int(temp),
            'wind_speed': wind_speed,
            'forecast': period.get('shortForecast', '')[:50],
            'data_source': 'NOAA'
        })
    
    return pd.DataFrame(weather_records)


def extract_weather_data_test(s3_client, bucket_name: str) -> str:
    """
    Test version of NOAA weather data extraction
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        
    Returns:
        S3 path to uploaded data
    """
    print("Extracting NOAA weather data...")
    
    # NOAA API endpoint
    url = "https://api.weather.gov/gridpoints/MTR/90,112/forecast"
    headers = {'User-Agent': 'GridSource Energy Bank Testing System'}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Transform data
    df = transform_weather_data(data)
    print(f"Processed {len(df)} weather forecast periods")
    
    # Upload to S3
    file_key = f'raw/weather/weather_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_buffer = df.to_csv(index=False)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_buffer
    )
    
    print(f"Uploaded weather data to s3://{bucket_name}/{file_key}")
    return f's3://{bucket_name}/{file_key}'


def transform_fred_data(api_data: Dict, indicator_name: str) -> pd.DataFrame:
    """Transform FRED API data to DataFrame"""
    economic_records = []
    
    for observation in api_data['observations']:
        if observation['value'] != '.':  # Skip missing values
            economic_records.append({
                'date': observation['date'],
                'indicator': indicator_name,
                'value': float(observation['value']),
                'data_source': 'FRED'
            })
    
    return pd.DataFrame(economic_records)


def extract_fred_data_test(api_key: str, s3_client, bucket_name: str) -> str:
    """
    Test version of FRED economic data extraction
    
    Args:
        api_key: FRED API key
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        
    Returns:
        S3 path to uploaded data
    """
    print("Extracting FRED economic data...")
    
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Economic indicators
    indicators = {
        'INDPRO': 'industrial_production_index',
        'CAUR': 'california_unemployment_rate',
        'DCOILWTICO': 'crude_oil_price_wti',
        'GASREGW': 'gasoline_price_regular'
    }
    
    all_data = []
    
    for series_id, indicator_name in indicators.items():
        print(f"Fetching {indicator_name} data...")
        
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'limit': 60,
            'sort_order': 'desc'
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = transform_fred_data(data, indicator_name)
            all_data.append(df)
            
        except Exception as e:
            print(f"Error fetching {indicator_name}: {str(e)}")
            continue
    
    # Combine all indicators
    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"Collected {len(final_df)} economic indicator records")
    
    # Upload to S3
    file_key = f'raw/economic/economic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_buffer = final_df.to_csv(index=False)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_buffer
    )
    
    print(f"Uploaded economic data to s3://{bucket_name}/{file_key}")
    return f's3://{bucket_name}/{file_key}'


def generate_energy_prices_test(s3_client, bucket_name: str, days: int = 30) -> str:
    """
    Test version of energy price data generation
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        days: Number of days of data to generate
        
    Returns:
        S3 path to uploaded data
    """
    print(f"Generating {days} days of energy price data...")
    
    # Generate date range
    end_date = datetime.now()
    dates = pd.date_range(start=end_date - timedelta(days=days), end=end_date, freq='D')
    
    # Simulate realistic energy prices
    base_price = 50  # $/MWh
    price_data = []
    
    for date in dates:
        # Add realistic price variation
        daily_variation = np.random.normal(0, 5)
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        price_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price_type': 'wholesale_electricity',
            'price_per_mwh': round(base_price * seasonal_factor + daily_variation, 2),
            'market': 'CAISO',
            'data_source': 'SIMULATED'
        })
    
    df = pd.DataFrame(price_data)
    print(f"Generated {len(df)} energy price records")
    
    # Upload to S3
    file_key = f'raw/prices/price_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_buffer = df.to_csv(index=False)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_buffer
    )
    
    print(f"Uploaded price data to s3://{bucket_name}/{file_key}")
    return f's3://{bucket_name}/{file_key}'


def create_ml_features_test(eia_df: pd.DataFrame, weather_df: pd.DataFrame, 
                           economic_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test version of ML feature creation
    
    Args:
        eia_df: EIA electricity data
        weather_df: Weather data
        economic_df: Economic indicators data
        price_df: Energy price data
        
    Returns:
        ML features DataFrame
    """
    print("Creating ML features from raw data...")
    
    # Aggregate EIA data by date
    eia_agg = eia_df.groupby('date').agg({
        'generation_mwh': 'sum'
    }).reset_index()
    eia_agg.columns = ['date', 'total_generation_mwh']
    
    # Average weather by date
    weather_agg = weather_df.groupby('date').agg({
        'temperature_f': 'mean'
    }).reset_index()
    weather_agg.columns = ['date', 'avg_temperature_f']
    
    # Pivot economic indicators
    economic_pivot = economic_df.pivot_table(
        index='date', 
        columns='indicator', 
        values='value', 
        aggfunc='first'
    ).reset_index()
    
    # Average prices by date
    price_agg = price_df.groupby('date').agg({
        'price_per_mwh': 'mean'
    }).reset_index()
    price_agg.columns = ['date', 'avg_electricity_price']
    
    # Merge all data
    features = eia_agg.copy()
    features = features.merge(weather_agg, on='date', how='left')
    features = features.merge(economic_pivot, on='date', how='left')
    features = features.merge(price_agg, on='date', how='left')
    
    # Create target variable (simplified liquidity calculation)
    features['liquidity_need_millions'] = (
        features['total_generation_mwh'] * 0.05 +
        features.get('crude_oil_price_wti', 50) * 10 +
        np.random.uniform(50, 200, len(features))
    )
    
    print(f"Created {len(features)} feature records")
    return features