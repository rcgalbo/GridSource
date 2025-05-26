-- Power BI Optimized Views for GridSource Bank Dashboard
-- These views are designed for optimal performance in Power BI

USE DATABASE GRID_SOURCE_BANK;
USE SCHEMA ENERGY_DATA;

-- =============================================================================
-- Executive Dashboard Views
-- =============================================================================

-- Main KPI Dashboard View - Current Status and Key Metrics
CREATE OR REPLACE VIEW v_executive_kpis AS
WITH current_status AS (
    SELECT 
        MAX(date) as latest_data_date,
        COUNT(*) as total_records_30d
    FROM ml_features 
    WHERE date >= CURRENT_DATE() - 30
),
latest_prediction AS (
    SELECT 
        date as prediction_date,
        predicted_liquidity_millions,
        model_version
    FROM ml_predictions 
    WHERE date >= CURRENT_DATE()
    ORDER BY date ASC
    LIMIT 1
),
recent_accuracy AS (
    SELECT 
        AVG(ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions)) as avg_error,
        COUNT(*) as prediction_count
    FROM ml_features mf
    JOIN ml_predictions mp ON mf.date = mp.date
    WHERE mf.date >= CURRENT_DATE() - 7
)
SELECT 
    cs.latest_data_date,
    cs.total_records_30d,
    lp.prediction_date,
    lp.predicted_liquidity_millions as next_day_liquidity_forecast,
    lp.model_version as current_model_version,
    ra.avg_error as recent_prediction_error_millions,
    ra.prediction_count as recent_predictions_evaluated,
    CASE 
        WHEN ra.avg_error <= 10 THEN 'Excellent'
        WHEN ra.avg_error <= 25 THEN 'Good'
        WHEN ra.avg_error <= 50 THEN 'Acceptable'
        ELSE 'Needs Attention'
    END as model_performance_status
FROM current_status cs
CROSS JOIN latest_prediction lp
CROSS JOIN recent_accuracy ra;

-- Time Series View for Line Charts
CREATE OR REPLACE VIEW v_liquidity_time_series AS
SELECT 
    mf.date,
    mf.liquidity_need_millions as actual_liquidity,
    mp.predicted_liquidity_millions as predicted_liquidity,
    ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) as prediction_error,
    mf.total_generation_mwh,
    mf.avg_temperature_f,
    mf.oil_price_usd,
    mf.avg_electricity_price,
    mp.model_version,
    -- Performance categories
    CASE 
        WHEN mp.predicted_liquidity_millions IS NULL THEN 'No Prediction'
        WHEN ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) <= 10 THEN 'High Accuracy'
        WHEN ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) <= 25 THEN 'Medium Accuracy'
        ELSE 'Low Accuracy'
    END as accuracy_category,
    -- Risk levels based on liquidity needs
    CASE 
        WHEN mf.liquidity_need_millions <= 100 THEN 'Low Risk'
        WHEN mf.liquidity_need_millions <= 200 THEN 'Medium Risk'
        ELSE 'High Risk'
    END as risk_level
FROM ml_features mf
LEFT JOIN ml_predictions mp ON mf.date = mp.date
WHERE mf.date >= CURRENT_DATE() - 90  -- Last 3 months
ORDER BY mf.date DESC;

-- =============================================================================
-- Energy Market Analysis Views  
-- =============================================================================

-- Daily Energy Generation by Fuel Type
CREATE OR REPLACE VIEW v_daily_generation_by_fuel AS
SELECT 
    date,
    fuel_type,
    SUM(generation_mwh) as total_generation_mwh,
    -- Calculate percentage of total daily generation
    SUM(generation_mwh) / SUM(SUM(generation_mwh)) OVER (PARTITION BY date) * 100 as generation_percentage
FROM raw_eia_data
WHERE date >= CURRENT_DATE() - 60  -- Last 2 months
GROUP BY date, fuel_type
ORDER BY date DESC, total_generation_mwh DESC;

-- Weather vs Energy Demand Analysis
CREATE OR REPLACE VIEW v_weather_energy_correlation AS
SELECT 
    mf.date,
    mf.avg_temperature_f,
    mf.total_generation_mwh,
    mf.liquidity_need_millions,
    w.wind_speed,
    w.forecast,
    -- Temperature categories for grouping
    CASE 
        WHEN mf.avg_temperature_f <= 60 THEN 'Cool (≤60°F)'
        WHEN mf.avg_temperature_f <= 75 THEN 'Mild (61-75°F)'
        WHEN mf.avg_temperature_f <= 85 THEN 'Warm (76-85°F)'
        ELSE 'Hot (>85°F)'
    END as temperature_category,
    -- Generation levels
    CASE 
        WHEN mf.total_generation_mwh <= 50000 THEN 'Low Generation'
        WHEN mf.total_generation_mwh <= 100000 THEN 'Medium Generation'
        ELSE 'High Generation'
    END as generation_level
FROM ml_features mf
LEFT JOIN raw_weather_data w ON mf.date = w.date
WHERE mf.date >= CURRENT_DATE() - 60
ORDER BY mf.date DESC;

-- Economic Indicators Impact View
CREATE OR REPLACE VIEW v_economic_indicators_impact AS
SELECT 
    mf.date,
    mf.liquidity_need_millions,
    mf.oil_price_usd,
    mf.industrial_production_index,
    mf.ca_unemployment_rate,
    -- Economic condition categories
    CASE 
        WHEN mf.oil_price_usd <= 60 THEN 'Low Oil Price'
        WHEN mf.oil_price_usd <= 80 THEN 'Medium Oil Price'
        ELSE 'High Oil Price'
    END as oil_price_category,
    CASE 
        WHEN mf.ca_unemployment_rate <= 4 THEN 'Low Unemployment'
        WHEN mf.ca_unemployment_rate <= 6 THEN 'Medium Unemployment'
        ELSE 'High Unemployment'
    END as unemployment_category
FROM ml_features mf
WHERE mf.date >= CURRENT_DATE() - 90
  AND mf.oil_price_usd IS NOT NULL
  AND mf.ca_unemployment_rate IS NOT NULL
ORDER BY mf.date DESC;

-- =============================================================================
-- Model Performance and Risk Views
-- =============================================================================

-- Model Performance Over Time
CREATE OR REPLACE VIEW v_model_performance_trend AS
WITH daily_errors AS (
    SELECT 
        mf.date,
        mp.model_version,
        ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions) as absolute_error,
        ((mf.liquidity_need_millions - mp.predicted_liquidity_millions) / mf.liquidity_need_millions) * 100 as percentage_error,
        mf.liquidity_need_millions as actual_value,
        mp.predicted_liquidity_millions as predicted_value
    FROM ml_features mf
    JOIN ml_predictions mp ON mf.date = mp.date
    WHERE mf.date >= CURRENT_DATE() - 60
),
weekly_performance AS (
    SELECT 
        DATE_TRUNC('WEEK', date) as week_start,
        model_version,
        AVG(absolute_error) as avg_mae,
        SQRT(AVG(absolute_error * absolute_error)) as rmse,
        AVG(ABS(percentage_error)) as avg_mape,
        COUNT(*) as prediction_count,
        MIN(actual_value) as min_actual,
        MAX(actual_value) as max_actual
    FROM daily_errors
    GROUP BY DATE_TRUNC('WEEK', date), model_version
)
SELECT 
    week_start,
    model_version,
    avg_mae,
    rmse,
    avg_mape,
    prediction_count,
    min_actual,
    max_actual,
    -- Performance rating
    CASE 
        WHEN avg_mae <= 15 THEN 'Excellent'
        WHEN avg_mae <= 30 THEN 'Good'
        WHEN avg_mae <= 50 THEN 'Acceptable'
        ELSE 'Poor'
    END as performance_rating
FROM weekly_performance
ORDER BY week_start DESC;

-- Risk Analysis View
CREATE OR REPLACE VIEW v_risk_analysis AS
WITH risk_metrics AS (
    SELECT 
        date,
        liquidity_need_millions,
        predicted_liquidity_millions,
        ABS(liquidity_need_millions - predicted_liquidity_millions) as prediction_error,
        -- Risk factors
        CASE WHEN liquidity_need_millions > 200 THEN 1 ELSE 0 END as high_liquidity_risk,
        CASE WHEN ABS(liquidity_need_millions - predicted_liquidity_millions) > 50 THEN 1 ELSE 0 END as high_prediction_risk,
        CASE WHEN oil_price_usd > 80 THEN 1 ELSE 0 END as oil_price_risk,
        CASE WHEN total_generation_mwh < 50000 THEN 1 ELSE 0 END as low_generation_risk
    FROM v_liquidity_time_series
    WHERE predicted_liquidity_millions IS NOT NULL
)
SELECT 
    date,
    liquidity_need_millions,
    predicted_liquidity_millions,
    prediction_error,
    high_liquidity_risk + high_prediction_risk + oil_price_risk + low_generation_risk as total_risk_score,
    CASE 
        WHEN high_liquidity_risk + high_prediction_risk + oil_price_risk + low_generation_risk = 0 THEN 'Low Risk'
        WHEN high_liquidity_risk + high_prediction_risk + oil_price_risk + low_generation_risk <= 2 THEN 'Medium Risk'
        ELSE 'High Risk'
    END as overall_risk_level,
    high_liquidity_risk,
    high_prediction_risk, 
    oil_price_risk,
    low_generation_risk
FROM risk_metrics
ORDER BY date DESC;

-- =============================================================================
-- Summary Statistics for Cards and KPIs
-- =============================================================================

-- Current Month Summary
CREATE OR REPLACE VIEW v_current_month_summary AS
SELECT 
    COUNT(*) as days_this_month,
    AVG(liquidity_need_millions) as avg_liquidity_need,
    MAX(liquidity_need_millions) as max_liquidity_need,
    MIN(liquidity_need_millions) as min_liquidity_need,
    AVG(total_generation_mwh) as avg_generation,
    AVG(avg_temperature_f) as avg_temperature,
    AVG(oil_price_usd) as avg_oil_price
FROM ml_features
WHERE DATE_TRUNC('MONTH', date) = DATE_TRUNC('MONTH', CURRENT_DATE());

-- Performance Summary for Current Model
CREATE OR REPLACE VIEW v_current_model_performance AS
WITH latest_model AS (
    SELECT model_version 
    FROM ml_predictions 
    ORDER BY prediction_timestamp DESC 
    LIMIT 1
),
model_stats AS (
    SELECT 
        mp.model_version,
        COUNT(*) as total_predictions,
        AVG(ABS(mf.liquidity_need_millions - mp.predicted_liquidity_millions)) as mae,
        SQRT(AVG(POWER(mf.liquidity_need_millions - mp.predicted_liquidity_millions, 2))) as rmse,
        CORR(mf.liquidity_need_millions, mp.predicted_liquidity_millions) as correlation
    FROM ml_predictions mp
    JOIN ml_features mf ON mp.date = mf.date
    JOIN latest_model lm ON mp.model_version = lm.model_version
    GROUP BY mp.model_version
)
SELECT 
    model_version,
    total_predictions,
    mae,
    rmse,
    correlation,
    CASE 
        WHEN mae <= 15 THEN 'Excellent'
        WHEN mae <= 30 THEN 'Good' 
        WHEN mae <= 50 THEN 'Acceptable'
        ELSE 'Needs Improvement'
    END as performance_grade
FROM model_stats;

-- Grant select permissions to Power BI service account
-- GRANT SELECT ON ALL VIEWS IN SCHEMA ENERGY_DATA TO ROLE POWERBI_READER;