"""
GridSource Bank Energy Banking Liquidity Forecasting Pipeline

This DAG implements an end-to-end data pipeline that:
1. Extracts data from 4 external sources (EIA, NOAA, FRED, Energy Prices)
2. Loads raw data into Snowflake data warehouse
3. Transforms data for machine learning
4. Trains ML model in SageMaker
5. Generates predictions and loads back to Snowflake

Data Flow: External APIs → S3 → Snowflake → SageMaker → Snowflake → Power BI
"""

from airflow import DAG
from airflow.decorators import task, dag
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import boto3
from typing import Dict, List

# DAG Configuration
default_args = {
    'owner': 'grid-source-bank',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 23),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Constants for configuration
S3_BUCKET = 'grid-source-data'
SNOWFLAKE_CONN_ID = 'snowflake_default'


@dag(
    dag_id='grid_source_automated_pipeline',
    default_args=default_args,
    description='Automated Grid Source Bank Energy Banking Pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['energy', 'banking', 'ml', 'forecasting']
)
def grid_source_pipeline():
    
    @task
    def extract_eia_electricity_data() -> str:
        """
        Extract electricity generation data from EIA API
        
        Returns:
            str: S3 path to uploaded CSV file
        """
        print("Starting EIA electricity data extraction...")
        
        # EIA API configuration
        api_key = "{{ var.value.eia_api_key }}"
        base_url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        
        # API parameters for California electricity data
        params = {
            'api_key': api_key,
            'frequency': 'daily',
            'data[0]': 'value',
            'facets[respondent][]': 'CAL',  # California
            'facets[fueltype][]': ['NG', 'SUN', 'WAT', 'WND'],  # Natural Gas, Solar, Water, Wind
            'start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d'),
            'sort[0][column]': 'period',
            'sort[0][direction]': 'desc',
            'length': 1000
        }
        
        # Make API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"Retrieved {len(data['response']['data'])} records from EIA")
        
        # Transform to DataFrame
        df = pd.DataFrame(data['response']['data'])
        df['date'] = pd.to_datetime(df['period'])
        df['generation_mwh'] = pd.to_numeric(df['value'], errors='coerce')
        df['fuel_type'] = df['fueltype']
        df['data_source'] = 'EIA'
        
        # Select and clean columns
        final_df = df[['date', 'fuel_type', 'generation_mwh', 'data_source']].dropna()
        
        # Upload to S3
        s3_client = boto3.client('s3')
        file_key = f'raw/eia/eia_data_{datetime.now().strftime("%Y%m%d")}.csv'
        csv_buffer = final_df.to_csv(index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=csv_buffer
        )
        
        print(f"Uploaded EIA data to s3://{S3_BUCKET}/{file_key}")
        return f's3://{S3_BUCKET}/{file_key}'
    
    @task
    def extract_weather_data() -> str:
        """
        Extract weather forecast data from NOAA API
        
        Returns:
            str: S3 path to uploaded CSV file
        """
        print("Starting NOAA weather data extraction...")
        
        # NOAA API endpoint for San Francisco Bay Area weather
        url = "https://api.weather.gov/gridpoints/MTR/90,112/forecast"
        
        headers = {
            'User-Agent': 'GridSource Energy Bank Forecasting System'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract weather periods
        weather_records = []
        for period in data['properties']['periods']:
            # Extract temperature (handle both F and C)
            temp = period.get('temperature', 0)
            
            # Parse wind speed (handle various formats like "10 mph")
            wind_speed_raw = period.get('windSpeed', '0 mph')
            wind_speed = ''.join(filter(str.isdigit, wind_speed_raw)) or '0'
            
            weather_records.append({
                'date': period['startTime'][:10],  # Extract date part only
                'temperature_f': int(temp),
                'wind_speed': int(wind_speed),
                'forecast': period.get('shortForecast', '')[:50],  # Limit length
                'data_source': 'NOAA'
            })
        
        df = pd.DataFrame(weather_records)
        print(f"Processed {len(df)} weather forecast periods")
        
        # Upload to S3
        s3_client = boto3.client('s3')
        file_key = f'raw/weather/weather_data_{datetime.now().strftime("%Y%m%d")}.csv'
        csv_buffer = df.to_csv(index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=csv_buffer
        )
        
        print(f"Uploaded weather data to s3://{S3_BUCKET}/{file_key}")
        return f's3://{S3_BUCKET}/{file_key}'
    
    @task
    def extract_economic_indicators() -> str:
        """
        Extract economic indicators from FRED API
        
        Returns:
            str: S3 path to uploaded CSV file
        """
        print("Starting FRED economic data extraction...")
        
        api_key = "{{ var.value.fred_api_key }}"
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Economic indicators relevant to energy banking
        indicators = {
            'INDPRO': 'industrial_production_index',
            'CAUR': 'california_unemployment_rate', 
            'DCOILWTICO': 'crude_oil_price_wti',
            'GASREGW': 'gasoline_price_regular'
        }
        
        all_economic_data = []
        
        for series_id, indicator_name in indicators.items():
            print(f"Fetching {indicator_name} data...")
            
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json',
                'limit': 60,  # ~2 months of data
                'sort_order': 'desc'
            }
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for observation in data['observations']:
                    if observation['value'] != '.':  # Skip missing values
                        all_economic_data.append({
                            'date': observation['date'],
                            'indicator': indicator_name,
                            'value': float(observation['value']),
                            'data_source': 'FRED'
                        })
                        
            except Exception as e:
                print(f"Error fetching {indicator_name}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_economic_data)
        print(f"Collected {len(df)} economic indicator records")
        
        # Upload to S3
        s3_client = boto3.client('s3')
        file_key = f'raw/economic/economic_data_{datetime.now().strftime("%Y%m%d")}.csv'
        csv_buffer = df.to_csv(index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=csv_buffer
        )
        
        print(f"Uploaded economic data to s3://{S3_BUCKET}/{file_key}")
        return f's3://{S3_BUCKET}/{file_key}'
    
    @task
    def extract_energy_prices() -> str:
        """
        Extract energy price data (simulated for demo)
        In production, this would connect to energy trading APIs
        
        Returns:
            str: S3 path to uploaded CSV file
        """
        print("Generating sample energy price data...")
        
        # Generate sample energy price data for demonstration
        import numpy as np
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='D')
        
        # Simulate realistic energy prices with some volatility
        base_price = 50  # $/MWh
        price_data = []
        
        for date in dates:
            # Add some realistic price variation
            daily_variation = np.random.normal(0, 5)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            price_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price_type': 'wholesale_electricity',
                'price_per_mwh': round(base_price * seasonal_factor + daily_variation, 2),
                'market': 'CAISO',  # California ISO
                'data_source': 'SIMULATED'
            })
        
        df = pd.DataFrame(price_data)
        print(f"Generated {len(df)} energy price records")
        
        # Upload to S3
        s3_client = boto3.client('s3')
        file_key = f'raw/prices/price_data_{datetime.now().strftime("%Y%m%d")}.csv'
        csv_buffer = df.to_csv(index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=csv_buffer
        )
        
        print(f"Uploaded energy price data to s3://{S3_BUCKET}/{file_key}")
        return f's3://{S3_BUCKET}/{file_key}'
    
    # Snowflake Data Loading Operations
    load_eia_data = SnowflakeOperator(
        task_id='load_eia_to_snowflake',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        COPY INTO ENERGY_DATA.raw_eia_data (date, fuel_type, generation_mwh, data_source)
        FROM @ENERGY_DATA.s3_stage/raw/eia/
        PATTERN = 'eia_data_{{ ds_nodash }}.csv'
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"')
        ON_ERROR = 'CONTINUE'
        PURGE = TRUE;
        """,
    )
    
    load_weather_data = SnowflakeOperator(
        task_id='load_weather_to_snowflake',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        COPY INTO ENERGY_DATA.raw_weather_data (date, temperature_f, wind_speed, forecast, data_source)
        FROM @ENERGY_DATA.s3_stage/raw/weather/
        PATTERN = 'weather_data_{{ ds_nodash }}.csv'
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"')
        ON_ERROR = 'CONTINUE'
        PURGE = TRUE;
        """,
    )
    
    load_economic_data = SnowflakeOperator(
        task_id='load_economic_to_snowflake',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        COPY INTO ENERGY_DATA.raw_economic_data (date, indicator, value, data_source)
        FROM @ENERGY_DATA.s3_stage/raw/economic/
        PATTERN = 'economic_data_{{ ds_nodash }}.csv'
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"')
        ON_ERROR = 'CONTINUE'
        PURGE = TRUE;
        """,
    )
    
    load_price_data = SnowflakeOperator(
        task_id='load_prices_to_snowflake',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        COPY INTO ENERGY_DATA.raw_price_data (date, price_type, price_per_mwh, market, data_source)
        FROM @ENERGY_DATA.s3_stage/raw/prices/
        PATTERN = 'price_data_{{ ds_nodash }}.csv'
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"')
        ON_ERROR = 'CONTINUE'
        PURGE = TRUE;
        """,
    )
    
    # Data Transformation for ML
    transform_data_for_ml = SnowflakeOperator(
        task_id='transform_data_for_ml',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        CREATE OR REPLACE TABLE ENERGY_DATA.ml_features AS
        SELECT 
            e.date,
            SUM(e.generation_mwh) as total_generation_mwh,
            AVG(w.temperature_f) as avg_temperature_f,
            MAX(CASE WHEN ec.indicator = 'crude_oil_price_wti' THEN ec.value END) as oil_price_usd,
            MAX(CASE WHEN ec.indicator = 'industrial_production_index' THEN ec.value END) as industrial_production_index,
            MAX(CASE WHEN ec.indicator = 'california_unemployment_rate' THEN ec.value END) as ca_unemployment_rate,
            AVG(p.price_per_mwh) as avg_electricity_price,
            -- Target variable: Simulated liquidity need based on generation and market factors
            (SUM(e.generation_mwh) * 0.05 + 
             COALESCE(MAX(CASE WHEN ec.indicator = 'crude_oil_price_wti' THEN ec.value END), 50) * 10 + 
             UNIFORM(50, 200, RANDOM())) as liquidity_need_millions
        FROM ENERGY_DATA.raw_eia_data e
        LEFT JOIN ENERGY_DATA.raw_weather_data w ON e.date = w.date
        LEFT JOIN ENERGY_DATA.raw_economic_data ec ON e.date = ec.date
        LEFT JOIN ENERGY_DATA.raw_price_data p ON e.date = p.date
        WHERE e.date IS NOT NULL 
          AND e.generation_mwh IS NOT NULL
        GROUP BY e.date
        HAVING total_generation_mwh > 0
        ORDER BY e.date DESC;
        """,
    )
    
    # Export training data for SageMaker
    export_training_data = SnowflakeOperator(
        task_id='export_training_data_to_s3',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        COPY INTO @ENERGY_DATA.s3_stage/ml/training_data.csv
        FROM (
            SELECT 
                total_generation_mwh,
                avg_temperature_f,
                oil_price_usd,
                industrial_production_index,
                avg_electricity_price,
                liquidity_need_millions
            FROM ENERGY_DATA.ml_features 
            WHERE total_generation_mwh IS NOT NULL 
              AND avg_temperature_f IS NOT NULL 
              AND oil_price_usd IS NOT NULL
              AND liquidity_need_millions IS NOT NULL
            ORDER BY date DESC
            LIMIT 1000
        )
        FILE_FORMAT = (TYPE = 'CSV' HEADER = TRUE)
        OVERWRITE = TRUE;
        """,
    )
    
    # SageMaker Model Training
    train_liquidity_model = SageMakerTrainingOperator(
        task_id='train_liquidity_forecasting_model',
        config={
            'TrainingJobName': 'grid-source-liquidity-{{ ds_nodash }}-{{ ts_nodash }}',
            'AlgorithmSpecification': {
                'TrainingImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                'TrainingInputMode': 'File'
            },
            'RoleArn': '{{ var.value.sagemaker_execution_role }}',
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{S3_BUCKET}/ml/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{S3_BUCKET}/models/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 10
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600
            },
            'HyperParameters': {
                'model-type': 'linear-regression'
            }
        }
    )
    
    @task
    def generate_and_store_predictions():
        """
        Generate predictions using the trained model and store in Snowflake
        """
        print("Generating liquidity predictions...")
        
        # For demo purposes, generate simple predictions
        # In production, this would load the actual trained model
        import numpy as np
        
        dates = pd.date_range(start=datetime.now() + timedelta(days=1),
                             end=datetime.now() + timedelta(days=7), freq='D')
        
        predictions = []
        for date in dates:
            # Simple prediction logic for demo
            base_liquidity = 150  # Million USD
            random_factor = np.random.normal(0, 20)
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_liquidity_millions': round(base_liquidity + random_factor, 2),
                'model_version': f'v{datetime.now().strftime("%Y%m%d")}',
                'prediction_timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(predictions)
        
        # Upload predictions to S3
        s3_client = boto3.client('s3')
        file_key = f'predictions/predictions_{datetime.now().strftime("%Y%m%d")}.csv'
        csv_buffer = df.to_csv(index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=csv_buffer
        )
        
        print(f"Generated {len(predictions)} predictions and uploaded to S3")
        return f's3://{S3_BUCKET}/{file_key}'
    
    # Load predictions back to Snowflake
    load_predictions = SnowflakeOperator(
        task_id='load_predictions_to_snowflake',
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
        COPY INTO ENERGY_DATA.ml_predictions (date, predicted_liquidity_millions, model_version, prediction_timestamp)
        FROM @ENERGY_DATA.s3_stage/predictions/
        PATTERN = 'predictions_{{ ds_nodash }}.csv'
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"')
        ON_ERROR = 'CONTINUE'
        PURGE = TRUE;
        """,
    )
    
    # Define task dependencies
    # 1. Extract data from all sources in parallel
    eia_extract = extract_eia_electricity_data()
    weather_extract = extract_weather_data()
    economic_extract = extract_economic_indicators()
    prices_extract = extract_energy_prices()
    
    # 2. Load all raw data to Snowflake in parallel
    raw_data_loads = [load_eia_data, load_weather_data, load_economic_data, load_price_data]
    
    # 3. Transform data for ML after all loads complete
    # 4. Export training data
    # 5. Train model
    # 6. Generate predictions
    # 7. Load predictions back to Snowflake
    
    [eia_extract, weather_extract, economic_extract, prices_extract] >> raw_data_loads
    raw_data_loads >> transform_data_for_ml >> export_training_data >> train_liquidity_model
    train_liquidity_model >> generate_and_store_predictions() >> load_predictions


# Instantiate the DAG
grid_source_dag = grid_source_pipeline()
