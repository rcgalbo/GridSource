# GridSource Bank - Energy Banking Liquidity Forecasting Pipeline

A comprehensive end-to-end data science pipeline for energy banking liquidity forecasting, demonstrating modern data engineering practices with AWS, Snowflake, and machine learning.

## 🎯 Project Overview

GridSource Bank implements an automated data pipeline that:
- **Extracts** data from 4 external sources (EIA, NOAA, FRED, Energy Markets)
- **Transforms** raw data into ML-ready features in Snowflake
- **Trains** predictive models using Amazon SageMaker
- **Delivers** insights through Power BI dashboards

**Data Flow**: `External APIs → S3 → Snowflake → SageMaker → Snowflake → Power BI`

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    EIA      │    │    NOAA     │    │    FRED     │    │   Energy    │
│ Electricity │    │   Weather   │    │  Economic   │    │   Prices    │
│    Data     │    │    Data     │    │ Indicators  │    │    Data     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       └──────────────────┼──────────────────┼──────────────────┘
                          │
                    ┌─────▼─────┐
                    │  Airflow  │
                    │    DAG    │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │     S3    │
                    │  Storage  │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ Snowflake │
                    │Data Warehouse│
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ SageMaker │
                    │ML Training│
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ Power BI  │
                    │Dashboard  │
                    └───────────┘
```

## 📁 Project Structure

```
GridSource/
├── airflow/          # Airflow DAGs and orchestration
├── research/         # Data exploration and API helpers
├── notebooks/        # Jupyter notebooks for analysis
├── tests/           # Testing framework (unit & integration)
├── sagemaker/       # ML training components
├── sql/             # Database schemas and views
├── config/          # Configuration files
└── docs/            # Documentation
```

## 🚀 Quick Start

### Prerequisites

- AWS Account with S3, SageMaker access
- Snowflake Account 
- Apache Airflow environment
- Power BI Pro/Premium (optional)
- API Keys: EIA, FRED

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd GridSource

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Configure Credentials

1. Update `config/airflow_variables.json` with your API keys and credentials
2. Import variables into Airflow:
   ```bash
   airflow variables import config/airflow_variables.json
   ```

### 3. Set Up Snowflake

```sql
-- Run the schema setup script
-- Execute: sql/snowflake_schema_setup.sql
```

### 4. Deploy Airflow DAG

```bash
# Copy DAG to Airflow dags folder
cp airflow/dags/grid_source_pipeline.py $AIRFLOW_HOME/dags/

# Trigger the DAG
airflow dags trigger grid_source_automated_pipeline
```

### 5. Set Up Power BI (Optional)

1. Connect Power BI to Snowflake using direct connection
2. Import the pre-built views from `sql/power_bi_views.sql`
3. Build dashboards using the provided view structure

## 📊 Data Sources

### 1. EIA (Energy Information Administration)
- **Purpose**: Electricity generation data by fuel type
- **API**: `https://api.eia.gov/v2/electricity/rto/daily-region-data/data/`
- **Coverage**: California electricity generation (Natural Gas, Solar, Hydro, Wind)
- **Frequency**: Daily

### 2. NOAA (National Weather Service)
- **Purpose**: Weather forecast data for energy demand correlation
- **API**: `https://api.weather.gov/gridpoints/MTR/90,112/forecast`
- **Coverage**: San Francisco Bay Area weather
- **Frequency**: Updated multiple times daily

### 3. FRED (Federal Reserve Economic Data)
- **Purpose**: Economic indicators affecting energy markets
- **API**: `https://api.stlouisfed.org/fred/series/observations`
- **Indicators**: 
  - Industrial Production Index
  - California Unemployment Rate
  - Crude Oil Prices (WTI)
  - Gasoline Prices
- **Frequency**: Monthly/Weekly

### 4. Energy Prices (Simulated)
- **Purpose**: Wholesale electricity prices
- **Market**: CAISO (California ISO)
- **Note**: Currently simulated data; replace with actual trading API

## 🤖 Machine Learning Model

### Features Used
- `total_generation_mwh`: Total electricity generation
- `avg_temperature_f`: Average temperature 
- `oil_price_usd`: Crude oil price
- `industrial_production_index`: Economic activity indicator
- `avg_electricity_price`: Average electricity price

### Target Variable
- `liquidity_need_millions`: Predicted liquidity requirement in millions USD

### Model Types Supported
- **Linear Regression**: Simple, interpretable baseline
- **Ridge Regression**: Regularized linear model
- **Random Forest**: Ensemble method for non-linear patterns

### Performance Metrics
- **MAE**: Mean Absolute Error (millions USD)
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 📈 Power BI Dashboards

### Executive Summary Dashboard
- Current liquidity need KPI
- Model prediction accuracy gauge
- 7-day liquidity forecast
- Risk level indicators

### Energy Market Analysis
- Electricity generation by fuel type (stacked area chart)
- Temperature vs energy demand correlation
- Economic indicators impact analysis

### Model Performance Monitoring
- Prediction accuracy over time
- Error distribution analysis
- Feature importance visualization
- Model drift detection

## 🔧 Configuration

Key configuration files:
- `config/settings.yaml`: Main project settings
- `config/airflow_variables.json`: Airflow variables template

### Important Settings

```yaml
# AWS Configuration
aws:
  s3:
    bucket_name: "grid-source-data"
  sagemaker:
    instance_type: "ml.m5.large"

# Snowflake Configuration  
snowflake:
  database: "GRID_SOURCE_BANK"
  schema: "ENERGY_DATA"

# ML Configuration
ml:
  training:
    test_size: 0.2
    random_state: 42
```

## 📝 Monitoring and Data Quality

### Automated Checks
- **Data Freshness**: Ensure data is no more than 2 days old
- **Completeness**: Verify all required columns have data
- **Volume**: Check minimum/maximum record counts
- **Accuracy**: Monitor prediction error trends

### Alerting
- Pipeline failures
- Data quality degradation  
- Model performance decline
- Prediction error spikes

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Solution: Implement exponential backoff in data extraction tasks

2. **Snowflake Connection Errors**
   - Check: Credentials, network connectivity, warehouse status

3. **SageMaker Training Failures**
   - Verify: IAM roles, S3 permissions, training data format

4. **Power BI Refresh Issues**
   - Check: Snowflake connection, data gateway status

### Monitoring Queries

```sql
-- Check data freshness
SELECT MAX(date) as latest_date, 
       DATEDIFF('day', MAX(date), CURRENT_DATE()) as days_behind
FROM ml_features;

-- Check model performance
SELECT AVG(ABS(liquidity_need_millions - predicted_liquidity_millions)) as avg_error
FROM v_liquidity_time_series 
WHERE date >= CURRENT_DATE() - 7;
```

## 🔐 Security Considerations

- All API keys stored as Airflow variables (encrypted)
- S3 server-side encryption enabled
- Snowflake SSL connections enforced
- Role-based access control implemented
- Audit logging enabled for all data access

## 📚 Learning Objectives

This project demonstrates:

### Data Engineering
- **ETL Pipeline Design**: Automated data extraction, transformation, loading
- **Cloud Data Architecture**: AWS S3, Snowflake, SageMaker integration
- **Workflow Orchestration**: Airflow DAG development and management

### Machine Learning
- **Feature Engineering**: Transform raw data into ML-ready features
- **Model Training**: SageMaker-based training pipeline
- **Model Monitoring**: Performance tracking and drift detection

### Business Intelligence
- **Data Visualization**: Power BI dashboard development
- **KPI Monitoring**: Real-time business metrics tracking
- **Self-Service Analytics**: Enabling business users with data tools

## 🎓 Next Steps for Learning

1. **Advanced ML**: Implement time series forecasting (ARIMA, Prophet)
2. **Real-time Processing**: Add streaming with Kinesis/Kafka
3. **MLOps**: Implement automated model retraining and deployment
4. **Advanced Analytics**: Add anomaly detection and predictive maintenance
5. **Cost Optimization**: Implement automated resource scaling

## 📞 Support

For questions or issues:
- Create an issue in the project repository
- Review the detailed setup guide in `docs/setup_guide.md`
- Check the troubleshooting section above

---

**Built with**: AWS, Snowflake, Apache Airflow, SageMaker, Power BI  
**Maintained by**: GridSource Bank Data Team  
**License**: MIT
