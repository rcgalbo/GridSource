-- GridSource Bank Snowflake Database Schema Setup
-- This script creates the complete database structure for the energy banking data pipeline

-- =============================================================================
-- Database and Schema Creation
-- =============================================================================

-- Create main database
CREATE DATABASE IF NOT EXISTS GRID_SOURCE_BANK;
USE DATABASE GRID_SOURCE_BANK;

-- Create schema for energy data
CREATE SCHEMA IF NOT EXISTS ENERGY_DATA;
USE SCHEMA ENERGY_DATA;

-- =============================================================================
-- Raw Data Tables - Store unprocessed data from external sources
-- =============================================================================

-- EIA Electricity Generation Data
CREATE TABLE IF NOT EXISTS raw_eia_data (
    date DATE NOT NULL,
    fuel_type VARCHAR(50) NOT NULL COMMENT 'Type of fuel used for generation (NG, SUN, WAT, WND)',
    generation_mwh DECIMAL(15,2) COMMENT 'Electricity generation in megawatt hours',
    data_source VARCHAR(20) DEFAULT 'EIA' COMMENT 'Source of the data',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_raw_eia PRIMARY KEY (date, fuel_type)
);

-- NOAA Weather Data
CREATE TABLE IF NOT EXISTS raw_weather_data (
    date DATE NOT NULL,
    temperature_f INTEGER COMMENT 'Temperature in Fahrenheit',
    wind_speed INTEGER COMMENT 'Wind speed in mph',
    forecast VARCHAR(100) COMMENT 'Short weather forecast description',
    data_source VARCHAR(20) DEFAULT 'NOAA' COMMENT 'Source of the data',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_raw_weather PRIMARY KEY (date)
);

-- FRED Economic Indicators
CREATE TABLE IF NOT EXISTS raw_economic_data (
    date DATE NOT NULL,
    indicator VARCHAR(100) NOT NULL COMMENT 'Economic indicator name',
    value DECIMAL(15,4) COMMENT 'Indicator value',
    data_source VARCHAR(20) DEFAULT 'FRED' COMMENT 'Source of the data',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_raw_economic PRIMARY KEY (date, indicator)
);

-- Energy Price Data
CREATE TABLE IF NOT EXISTS raw_price_data (
    date DATE NOT NULL,
    price_type VARCHAR(50) NOT NULL COMMENT 'Type of energy price',
    price_per_mwh DECIMAL(10,2) COMMENT 'Price per megawatt hour in USD',
    market VARCHAR(20) COMMENT 'Energy market (e.g., CAISO)',
    data_source VARCHAR(20) COMMENT 'Source of the data',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_raw_price PRIMARY KEY (date, price_type, market)
);

-- =============================================================================
-- Machine Learning Feature Table - Transformed data for model training
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_features (
    date DATE NOT NULL,
    total_generation_mwh DECIMAL(15,2) COMMENT 'Total electricity generation across all fuel types',
    avg_temperature_f DECIMAL(5,1) COMMENT 'Average temperature in Fahrenheit',
    oil_price_usd DECIMAL(8,2) COMMENT 'Crude oil price in USD per barrel',
    industrial_production_index DECIMAL(8,2) COMMENT 'Industrial production index value',
    ca_unemployment_rate DECIMAL(5,2) COMMENT 'California unemployment rate percentage',
    avg_electricity_price DECIMAL(8,2) COMMENT 'Average electricity price per MWh',
    liquidity_need_millions DECIMAL(12,2) COMMENT 'Target variable: Liquidity need in millions USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_ml_features PRIMARY KEY (date)
);

-- =============================================================================
-- Model Predictions Table - Store ML model outputs
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_predictions (
    date DATE NOT NULL COMMENT 'Prediction date',
    predicted_liquidity_millions DECIMAL(12,2) COMMENT 'Predicted liquidity need in millions USD',
    actual_liquidity_millions DECIMAL(12,2) COMMENT 'Actual liquidity need (populated later)',
    model_version VARCHAR(50) COMMENT 'Version identifier of the model used',
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'When prediction was made',
    confidence_interval_lower DECIMAL(12,2) COMMENT 'Lower bound of prediction confidence interval',
    confidence_interval_upper DECIMAL(12,2) COMMENT 'Upper bound of prediction confidence interval',
    CONSTRAINT pk_ml_predictions PRIMARY KEY (date, model_version)
);

-- =============================================================================
-- Model Performance Tracking - Monitor model accuracy over time
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_performance (
    evaluation_date DATE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    mae_millions DECIMAL(10,2) COMMENT 'Mean Absolute Error in millions USD',
    rmse_millions DECIMAL(10,2) COMMENT 'Root Mean Square Error in millions USD',
    r_squared DECIMAL(5,4) COMMENT 'R-squared coefficient of determination',
    mape_percent DECIMAL(5,2) COMMENT 'Mean Absolute Percentage Error',
    num_predictions INTEGER COMMENT 'Number of predictions evaluated',
    evaluation_period_days INTEGER COMMENT 'Period length for evaluation',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_model_performance PRIMARY KEY (evaluation_date, model_version)
);

-- =============================================================================
-- Data Quality Monitoring - Track data pipeline health
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_quality_metrics (
    check_date DATE NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL COMMENT 'e.g., row_count, null_percentage, freshness_hours',
    metric_value DECIMAL(15,4) COMMENT 'Numeric value of the metric',
    status VARCHAR(20) COMMENT 'PASS, WARN, FAIL',
    threshold_value DECIMAL(15,4) COMMENT 'Threshold that was checked against',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_data_quality PRIMARY KEY (check_date, table_name, metric_name)
);

-- =============================================================================
-- S3 Stage and File Format Setup
-- =============================================================================

-- Create S3 stage for data loading (update with your actual S3 credentials)
CREATE STAGE IF NOT EXISTS s3_stage 
    URL = 's3://grid-source-data/'
    CREDENTIALS = (
        AWS_KEY_ID = '{{ var.value.aws_access_key_id }}'
        AWS_SECRET_KEY = '{{ var.value.aws_secret_access_key }}'
    )
    COMMENT = 'S3 stage for GridSource data pipeline';

-- Create file format for CSV files
CREATE FILE FORMAT IF NOT EXISTS csv_format
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '')
    EMPTY_FIELD_AS_NULL = TRUE
    COMMENT = 'Standard CSV format for data loading';

-- =============================================================================
-- Views for Power BI and Analytics
-- =============================================================================

-- Daily Energy Dashboard View
CREATE OR REPLACE VIEW v_daily_energy_dashboard AS
SELECT 
    mf.date,
    mf.total_generation_mwh,
    mf.avg_temperature_f,
    mf.avg_electricity_price,
    mf.liquidity_need_millions as actual_liquidity,
    mp.predicted_liquidity_millions,
    ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) as prediction_error_millions,
    CASE 
        WHEN mp.predicted_liquidity_millions IS NULL THEN 'No Prediction'
        WHEN ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) <= 10 THEN 'Accurate'
        WHEN ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) <= 25 THEN 'Acceptable'
        ELSE 'Poor'
    END as prediction_quality
FROM ml_features mf
LEFT JOIN ml_predictions mp ON mf.date = mp.date 
    AND mp.model_version = (
        SELECT model_version 
        FROM ml_predictions 
        WHERE date = mf.date 
        ORDER BY prediction_timestamp DESC 
        LIMIT 1
    )
ORDER BY mf.date DESC;

-- Weekly Aggregated View
CREATE OR REPLACE VIEW v_weekly_summary AS
SELECT 
    DATE_TRUNC('WEEK', date) as week_start,
    COUNT(*) as days_in_week,
    AVG(total_generation_mwh) as avg_daily_generation,
    AVG(avg_temperature_f) as avg_temperature,
    AVG(liquidity_need_millions) as avg_liquidity_need,
    AVG(predicted_liquidity_millions) as avg_predicted_liquidity,
    AVG(ABS(liquidity_need_millions - predicted_liquidity_millions)) as avg_prediction_error
FROM v_daily_energy_dashboard
WHERE predicted_liquidity_millions IS NOT NULL
GROUP BY DATE_TRUNC('WEEK', date)
ORDER BY week_start DESC;

-- Current Status View for Real-time Dashboard
CREATE OR REPLACE VIEW v_current_status AS
SELECT 
    'Latest Data' as metric_category,
    MAX(date) as latest_date,
    DATEDIFF('day', MAX(date), CURRENT_DATE()) as days_behind
FROM ml_features
UNION ALL
SELECT 
    'Data Volume' as metric_category,
    COUNT(*)::DATE as latest_date,
    NULL as days_behind
FROM ml_features
WHERE date >= CURRENT_DATE() - 30
UNION ALL
SELECT 
    'Prediction Coverage' as metric_category,
    COUNT(*)::DATE as latest_date,
    NULL as days_behind
FROM ml_predictions
WHERE date >= CURRENT_DATE() - 7;

-- =============================================================================
-- Indexes for Performance Optimization
-- =============================================================================

-- Performance indexes on frequently queried columns
-- Note: Snowflake auto-clusters, but these hints help with query planning

-- Time-based clustering for time series queries
ALTER TABLE ml_features CLUSTER BY (date);
ALTER TABLE ml_predictions CLUSTER BY (date);
ALTER TABLE raw_eia_data CLUSTER BY (date);

-- =============================================================================
-- Data Retention Policies
-- =============================================================================

-- Set retention policy for raw data (keep 2 years)
ALTER TABLE raw_eia_data SET DATA_RETENTION_TIME_IN_DAYS = 730;
ALTER TABLE raw_weather_data SET DATA_RETENTION_TIME_IN_DAYS = 730;
ALTER TABLE raw_economic_data SET DATA_RETENTION_TIME_IN_DAYS = 730;
ALTER TABLE raw_price_data SET DATA_RETENTION_TIME_IN_DAYS = 730;

-- Set retention policy for ML data (keep 3 years)
ALTER TABLE ml_features SET DATA_RETENTION_TIME_IN_DAYS = 1095;
ALTER TABLE ml_predictions SET DATA_RETENTION_TIME_IN_DAYS = 1095;

-- =============================================================================
-- Sample Data Quality Checks (run these periodically)
-- =============================================================================

-- Insert sample data quality metrics
INSERT INTO data_quality_metrics (check_date, table_name, metric_name, metric_value, status, threshold_value)
SELECT 
    CURRENT_DATE() as check_date,
    'ml_features' as table_name,
    'row_count' as metric_name,
    COUNT(*) as metric_value,
    CASE WHEN COUNT(*) >= 30 THEN 'PASS' ELSE 'FAIL' END as status,
    30 as threshold_value
FROM ml_features
WHERE date >= CURRENT_DATE() - 30;

-- =============================================================================
-- Grants and Security (adjust based on your security requirements)
-- =============================================================================

-- Example role-based access (uncomment and modify as needed)
/*
-- Create roles for different user types
CREATE ROLE IF NOT EXISTS grid_source_reader;
CREATE ROLE IF NOT EXISTS grid_source_analyst;
CREATE ROLE IF NOT EXISTS grid_source_admin;

-- Grant permissions
GRANT USAGE ON DATABASE GRID_SOURCE_BANK TO ROLE grid_source_reader;
GRANT USAGE ON SCHEMA ENERGY_DATA TO ROLE grid_source_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA ENERGY_DATA TO ROLE grid_source_reader;
GRANT SELECT ON ALL VIEWS IN SCHEMA ENERGY_DATA TO ROLE grid_source_reader;

-- Analyst role (can read and insert predictions)
GRANT ROLE grid_source_reader TO ROLE grid_source_analyst;
GRANT INSERT ON TABLE ml_predictions TO ROLE grid_source_analyst;

-- Admin role (full access)
GRANT ALL ON DATABASE GRID_SOURCE_BANK TO ROLE grid_source_admin;
*/

-- =============================================================================
-- Setup Validation Query
-- =============================================================================

-- Run this to verify setup completed successfully
SELECT 
    'Database Setup Complete' as status,
    COUNT(*) as tables_created
FROM information_schema.tables 
WHERE table_schema = 'ENERGY_DATA';