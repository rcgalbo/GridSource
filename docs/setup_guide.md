# GridSource Bank - Detailed Setup Guide

This guide provides step-by-step instructions for setting up the complete GridSource Bank energy banking liquidity forecasting pipeline.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Setup](#aws-setup)
3. [Snowflake Setup](#snowflake-setup)
4. [Airflow Setup](#airflow-setup)
5. [SageMaker Configuration](#sagemaker-configuration)
6. [Power BI Setup](#power-bi-setup)
7. [Testing the Pipeline](#testing-the-pipeline)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Accounts Required
- **AWS Account** with billing enabled
- **Snowflake Account** (30-day trial available)
- **EIA API Key** (free registration)
- **FRED API Key** (free registration)
- **Power BI Pro** (optional, for dashboards)

### Software Requirements
- Python 3.8+
- Apache Airflow 2.0+
- AWS CLI
- Git

### API Keys Registration

#### 1. EIA API Key
1. Visit: https://www.eia.gov/opendata/register.php
2. Register for a free account
3. Copy your API key from the dashboard
4. **Save this key** - you'll need it later

#### 2. FRED API Key  
1. Visit: https://research.stlouisfed.org/useraccount/apikey
2. Create a free FRED account
3. Request an API key
4. **Save this key** - you'll need it later

## AWS Setup

### 1. Create S3 Bucket

```bash
# Install AWS CLI if not already installed
pip install awscli

# Configure AWS credentials
aws configure

# Create S3 bucket (replace with unique name)
aws s3 mb s3://grid-source-data-YOUR-UNIQUE-ID

# Create folder structure
aws s3api put-object --bucket grid-source-data-YOUR-UNIQUE-ID --key raw/
aws s3api put-object --bucket grid-source-data-YOUR-UNIQUE-ID --key processed/
aws s3api put-object --bucket grid-source-data-YOUR-UNIQUE-ID --key ml/
aws s3api put-object --bucket grid-source-data-YOUR-UNIQUE-ID --key models/
aws s3api put-object --bucket grid-source-data-YOUR-UNIQUE-ID --key predictions/
```

### 2. Create IAM Roles

#### SageMaker Execution Role
```bash
# Create trust policy file
cat > sagemaker-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name GridSourceSageMakerRole \
    --assume-role-policy-document file://sagemaker-trust-policy.json

# Attach required policies
aws iam attach-role-policy \
    --role-name GridSourceSageMakerRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name GridSourceSageMakerRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

#### Airflow S3 Access Policy
```bash
# Create S3 access policy
cat > airflow-s3-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::grid-source-data-YOUR-UNIQUE-ID",
                "arn:aws:s3:::grid-source-data-YOUR-UNIQUE-ID/*"
            ]
        }
    ]
}
EOF

# Create and attach policy to your user/role
aws iam create-policy \
    --policy-name GridSourceS3Access \
    --policy-document file://airflow-s3-policy.json
```

### 3. Test AWS Setup

```bash
# Test S3 access
aws s3 ls s3://grid-source-data-YOUR-UNIQUE-ID

# Verify SageMaker role
aws iam get-role --role-name GridSourceSageMakerRole
```

## Snowflake Setup

### 1. Create Snowflake Account
1. Visit: https://signup.snowflake.com/
2. Choose **AWS** as cloud provider
3. Select region closest to your location
4. Complete account setup

### 2. Initial Database Setup

Connect to Snowflake using web interface or SnowSQL:

```sql
-- Create main database and schema
CREATE DATABASE GRID_SOURCE_BANK;
USE DATABASE GRID_SOURCE_BANK;
CREATE SCHEMA ENERGY_DATA;
USE SCHEMA ENERGY_DATA;

-- Create a warehouse for compute
CREATE WAREHOUSE COMPUTE_WH WITH 
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE;

-- Use the warehouse
USE WAREHOUSE COMPUTE_WH;
```

### 3. Run Schema Setup Script

```sql
-- Execute the complete schema setup
-- Copy and paste contents of sql/snowflake_schema_setup.sql
-- This creates all tables, views, and stages
```

### 4. Configure S3 Integration

```sql
-- Create S3 stage (update with your credentials)
CREATE STAGE s3_stage 
    URL = 's3://grid-source-data-YOUR-UNIQUE-ID/'
    CREDENTIALS = (
        AWS_KEY_ID = 'YOUR_AWS_ACCESS_KEY_ID'
        AWS_SECRET_KEY = 'YOUR_AWS_SECRET_ACCESS_KEY'
    );

-- Test the stage
LIST @s3_stage;
```

### 5. Create Snowflake User for Airflow

```sql
-- Create service account for Airflow
CREATE USER airflow_user 
    PASSWORD = 'SecurePassword123!'
    DEFAULT_ROLE = 'ACCOUNTADMIN'
    DEFAULT_WAREHOUSE = 'COMPUTE_WH'
    DEFAULT_NAMESPACE = 'GRID_SOURCE_BANK.ENERGY_DATA';

-- Grant necessary permissions
GRANT ROLE ACCOUNTADMIN TO USER airflow_user;
```

## Airflow Setup

### 1. Install Airflow

```bash
# Create virtual environment
python -m venv airflow_env
source airflow_env/bin/activate  # On Windows: airflow_env\Scripts\activate

# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Install Airflow with required providers
pip install "apache-airflow[amazon,snowflake]==2.5.1" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.5.1/constraints-3.8.txt"
```

### 2. Initialize Airflow

```bash
# Initialize the database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname GridSource \
    --lastname Admin \
    --role Admin \
    --email admin@gridsourcebank.com \
    --password admin123
```

### 3. Configure Connections

#### Snowflake Connection
```bash
airflow connections add \
    --conn-id snowflake_default \
    --conn-type Snowflake \
    --conn-host YOUR_ACCOUNT.snowflakecomputing.com \
    --conn-login airflow_user \
    --conn-password SecurePassword123! \
    --conn-schema ENERGY_DATA \
    --conn-extra '{"database": "GRID_SOURCE_BANK", "warehouse": "COMPUTE_WH", "role": "ACCOUNTADMIN"}'
```

#### AWS Connection
```bash
airflow connections add \
    --conn-id aws_default \
    --conn-type Amazon Web Services \
    --conn-extra '{"aws_access_key_id": "YOUR_ACCESS_KEY", "aws_secret_access_key": "YOUR_SECRET_KEY", "region_name": "us-west-2"}'
```

### 4. Set Airflow Variables

```bash
# Update config/airflow_variables.json with your actual values
# Then import:
airflow variables import config/airflow_variables.json
```

### 5. Deploy DAG

```bash
# Copy DAG file to Airflow dags folder
cp airflow/dags/grid_source_pipeline.py $AIRFLOW_HOME/dags/

# Start Airflow services
airflow webserver --port 8080 &
airflow scheduler &
```

### 6. Access Airflow UI

1. Open browser to http://localhost:8080
2. Login with username: `admin`, password: `admin123`
3. Verify the `grid_source_automated_pipeline` DAG appears

## SageMaker Configuration

### 1. Upload Training Script

```bash
# Create SageMaker directory in S3
aws s3 cp sagemaker/train.py s3://grid-source-data-YOUR-UNIQUE-ID/sagemaker/train.py
aws s3 cp sagemaker/requirements.txt s3://grid-source-data-YOUR-UNIQUE-ID/sagemaker/requirements.txt
```

### 2. Test SageMaker Access

```python
import boto3

# Test SageMaker client
sagemaker = boto3.client('sagemaker', region_name='us-west-2')

# List training jobs (should return empty list initially)
response = sagemaker.list_training_jobs(MaxResults=10)
print("SageMaker connection successful:", response)
```

### 3. Verify IAM Role

```bash
# Get the role ARN for configuration
aws iam get-role --role-name GridSourceSageMakerRole --query 'Role.Arn'
```

Update the role ARN in your Airflow variables:
```json
{
  "sagemaker_execution_role": "arn:aws:iam::YOUR_ACCOUNT_ID:role/GridSourceSageMakerRole"
}
```

## Power BI Setup

### 1. Install Power BI Desktop
- Download from: https://powerbi.microsoft.com/desktop/
- Install and launch

### 2. Install Snowflake Connector
1. In Power BI Desktop, click **Get Data**
2. Search for **Snowflake**
3. Install connector if prompted

### 3. Connect to Snowflake

1. **Get Data** → **Snowflake**
2. Enter connection details:
   - **Server**: `YOUR_ACCOUNT.snowflakecomputing.com`
   - **Warehouse**: `COMPUTE_WH`
   - **Database**: `GRID_SOURCE_BANK`
3. **Authentication**: Username/Password
4. **Username**: `airflow_user`
5. **Password**: `SecurePassword123!`

### 4. Import Power BI Views

1. Connect to Snowflake
2. Navigate to `GRID_SOURCE_BANK` → `ENERGY_DATA`
3. Select views starting with `v_`:
   - `v_executive_kpis`
   - `v_liquidity_time_series`
   - `v_daily_generation_by_fuel`
   - `v_model_performance_trend`

### 5. Build Basic Dashboard

#### Executive KPI Card
1. Add **Card** visual
2. Field: `v_executive_kpis[next_day_liquidity_forecast]`
3. Title: "Next Day Liquidity Forecast"

#### Time Series Chart
1. Add **Line Chart**
2. X-axis: `v_liquidity_time_series[date]`
3. Y-axis: `v_liquidity_time_series[actual_liquidity]`
4. Second Y-axis: `v_liquidity_time_series[predicted_liquidity]`

## Testing the Pipeline

### 1. Test Individual Components

#### Test Data Extraction
```bash
# Trigger just the extraction tasks
airflow tasks test grid_source_automated_pipeline extract_eia_electricity_data 2025-05-23
```

#### Test Snowflake Connection
```sql
-- Verify tables exist
SHOW TABLES IN SCHEMA ENERGY_DATA;

-- Check for any sample data
SELECT COUNT(*) FROM raw_eia_data;
```

### 2. Run Full Pipeline

```bash
# Trigger the complete DAG
airflow dags trigger grid_source_automated_pipeline

# Monitor progress
airflow dags state grid_source_automated_pipeline 2025-05-23
```

### 3. Validate Data Flow

```sql
-- Check data in each stage
SELECT 'Raw EIA' as table_name, COUNT(*) as record_count FROM raw_eia_data
UNION ALL
SELECT 'Raw Weather', COUNT(*) FROM raw_weather_data
UNION ALL
SELECT 'Raw Economic', COUNT(*) FROM raw_economic_data
UNION ALL
SELECT 'ML Features', COUNT(*) FROM ml_features
UNION ALL
SELECT 'ML Predictions', COUNT(*) FROM ml_predictions;
```

### 4. Test Power BI Connection

1. Refresh data in Power BI Desktop
2. Verify charts populate with data
3. Test interactive filtering

## Troubleshooting

### Common Issues and Solutions

#### 1. Airflow DAG Not Appearing
```bash
# Check for syntax errors
python -m py_compile $AIRFLOW_HOME/dags/grid_source_pipeline.py

# Check Airflow logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

#### 2. Snowflake Connection Failed
- Verify account URL format: `account.region.snowflakecomputing.com`
- Check username/password
- Ensure warehouse is running
- Verify network connectivity

#### 3. S3 Access Denied
```bash
# Check bucket exists and permissions
aws s3 ls s3://grid-source-data-YOUR-UNIQUE-ID

# Verify IAM policies
aws iam list-attached-user-policies --user-name YOUR_USERNAME
```

#### 4. SageMaker Training Job Failed
- Check CloudWatch logs in AWS Console
- Verify S3 paths in training configuration
- Ensure IAM role has necessary permissions
- Check training data format

#### 5. API Rate Limits
- EIA API: 5,000 requests per hour
- FRED API: 120 requests per minute
- Implement exponential backoff if needed

### Monitoring Commands

```bash
# Check Airflow task status
airflow tasks list grid_source_automated_pipeline

# View task logs
airflow tasks log grid_source_automated_pipeline extract_eia_electricity_data 2025-05-23

# Check variable values
airflow variables list
```

### Performance Optimization

#### Snowflake
```sql
-- Monitor warehouse usage
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
WHERE WAREHOUSE_NAME = 'COMPUTE_WH'
ORDER BY START_TIME DESC;

-- Optimize clustering
ALTER TABLE ml_features CLUSTER BY (date);
```

#### Airflow
```python
# Increase parallelism in airflow.cfg
max_active_tasks_per_dag = 16
parallelism = 32
```

## Next Steps

Once the basic pipeline is working:

1. **Schedule Regular Runs**: Set DAG to run daily
2. **Monitor Performance**: Set up alerts for failures
3. **Optimize Costs**: Adjust warehouse sizes based on usage
4. **Enhance Models**: Add more sophisticated ML algorithms
5. **Expand Dashboards**: Build additional Power BI reports

## Security Checklist

- [ ] All API keys stored as Airflow variables (encrypted)
- [ ] S3 bucket has proper access controls
- [ ] Snowflake uses dedicated service account
- [ ] SageMaker execution role follows least privilege
- [ ] Power BI uses read-only database access
- [ ] Regular credential rotation scheduled

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review component-specific logs
3. Verify all prerequisites are met
4. Ensure all credentials are correctly configured

---

**Estimated Setup Time**: 4-6 hours for complete setup  
**Prerequisites**: AWS Account, Snowflake Account, API Keys  
**Difficulty**: Intermediate