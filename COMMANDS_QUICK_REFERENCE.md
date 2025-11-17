# 🚀 NYC 311 ML Pipeline - Quick Command Reference

## 🐳 Docker Commands

### Build & Start Services
```bash
# Build ML worker image
docker compose build ml-worker

# Start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs ml-worker
docker compose logs airflow-scheduler
```

### Stop & Restart
```bash
# Stop all services
docker compose down

# Restart specific service
docker compose restart ml-worker

# Rebuild and restart
docker compose build ml-worker && docker compose up -d ml-worker
```

## 🤖 ML Pipeline Commands

### Run Complete Pipeline
```bash
# Run all ML tasks (Classification + Forecasting + Regression)
docker compose exec ml-worker python /app/scripts/ml_pipeline.py
```

### Run Individual Models

#### Classification Models
```bash
docker compose exec ml-worker python /app/scripts/ml_classification.py
```

#### Time Series Forecasting
```bash
docker compose exec ml-worker python /app/scripts/ml_forecasting.py
```

#### Spark Regression
```bash
docker compose exec ml-worker python /app/scripts/spark_regression.py
```

## 📊 Database Commands

### Check Data
```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U airflow -d airflow

# Count records
docker compose exec postgres psql -U airflow -d airflow -c "SELECT COUNT(*) FROM nyc_311_bronx_full_year;"

# View forecast results
docker compose exec postgres psql -U airflow -d airflow -c "SELECT * FROM ml_forecast_results LIMIT 5;"

# Check recent complaints
docker compose exec postgres psql -U airflow -d airflow -c "SELECT complaint_type, COUNT(*) as count FROM nyc_311_bronx_full_year GROUP BY complaint_type ORDER BY count DESC LIMIT 10;"
```

## 📁 File Commands

### Check Generated Files
```bash
# List models
ls -lh /Users/sandilyachimalamarri/data-pipeline-project/models/

# List reports
ls -lh /Users/sandilyachimalamarri/data-pipeline-project/reports/

# View forecast plot
open /Users/sandilyachimalamarri/data-pipeline-project/reports/forecast_plot.png
```

### Access from Container
```bash
# List models from ml-worker
docker compose exec ml-worker ls -lh /app/models/

# View model info
docker compose exec ml-worker python -c "import joblib; model = joblib.load('/app/models/xgboost_model.pkl'); print(model)"
```

## 🌐 Access Web UIs

```bash
# Open Airflow (or visit http://localhost:8080)
open http://localhost:8080

# Open Metabase (or visit http://localhost:3000)
open http://localhost:3000

```

### Login Credentials

**Airflow:**
- Username: `airflow`
- Password: `airflow`

**Metabase:** (first-time setup required)
- Create admin account on first visit
- Database: `postgres:5432/airflow`
- DB User: `airflow`
- DB Password: `airflow`

## ⚡ Airflow DAG Commands

### From Airflow UI
1. Go to http://localhost:8080
2. Find `nyc_data_pipeline`
3. Click "Play" button to trigger

### From Command Line
```bash
# List DAGs
docker compose exec airflow-scheduler airflow dags list

# Trigger DAG
docker compose exec airflow-scheduler airflow dags trigger nyc_data_pipeline

# Check DAG status
docker compose exec airflow-scheduler airflow dags list-runs -d nyc_data_pipeline

# View task logs
docker compose exec airflow-scheduler airflow tasks logs nyc_data_pipeline ml_pipeline_orchestrator <execution_date>
```

## 🔍 Debugging Commands

### Check Container Status
```bash
# Check if ml-worker is running
docker compose ps ml-worker

# Get detailed info
docker inspect ml-worker

# Check resource usage
docker stats ml-worker
```

### View Logs
```bash
# Follow logs in real-time
docker compose logs -f ml-worker

# Last 100 lines
docker compose logs --tail=100 ml-worker

# Since last 10 minutes
docker compose logs --since=10m ml-worker
```

### Enter Container Shell
```bash
# Interactive shell in ml-worker
docker compose exec ml-worker bash

# Then inside container:
cd /app
ls -la models/
ls -la reports/
python
>>> import sklearn, xgboost, prophet
>>> print("All libraries loaded!")
```

### Test Python Environment
```bash
# Check Python version
docker compose exec ml-worker python --version

# Check installed packages
docker compose exec ml-worker pip list | grep -E "scikit-learn|xgboost|prophet|pyspark"

# Test imports
docker compose exec ml-worker python -c "import sklearn, xgboost, prophet, pyspark; print('✅ All ML libraries working!')"
```

## 🧹 Cleanup Commands

### Remove Old Models/Reports
```bash
rm -rf /Users/sandilyachimalamarri/data-pipeline-project/models/*
rm -rf /Users/sandilyachimalamarri/data-pipeline-project/reports/*
```

### Reset Database Tables
```bash
docker compose exec postgres psql -U airflow -d airflow -c "DROP TABLE IF EXISTS ml_forecast_results;"
```

### Full Reset
```bash
# Stop everything
docker compose down

# Remove volumes (WARNING: deletes all data!)
docker compose down -v

# Remove images
docker rmi data-pipeline-project-ml-worker

# Rebuild from scratch
docker compose build ml-worker
docker compose up -d
```

## 📈 Monitoring Commands

### Check Model Performance
```bash
# View classification accuracy
docker compose exec ml-worker python -c "
import joblib
import os
if os.path.exists('/app/models/xgboost_model.pkl'):
    print('✅ XGBoost model found')
else:
    print('❌ XGBoost model not found - run training first')
"

# Check forecast table
docker compose exec postgres psql -U airflow -d airflow -c "
SELECT COUNT(*) as forecast_rows,
       MIN(date) as earliest_forecast,
       MAX(date) as latest_forecast
FROM ml_forecast_results;
"
```

### Verify Data Pipeline
```bash
# Check data freshness
docker compose exec postgres psql -U airflow -d airflow -c "
SELECT 
    MAX(created_at) as latest_complaint,
    COUNT(*) as total_complaints,
    COUNT(DISTINCT DATE(created_at)) as days_of_data
FROM nyc_311_bronx_full_year;
"
```

## 🔄 Update & Maintenance

### Update ML Scripts
```bash
# After editing scripts, restart ml-worker
docker compose restart ml-worker

# Or rebuild if you changed requirements.txt
docker compose build ml-worker
docker compose up -d ml-worker
```

### Update Airflow DAG
```bash
# After editing DAG file
docker compose restart airflow-scheduler airflow-webserver

# DAG should refresh automatically in ~30 seconds
```

## 💡 Pro Tips

```bash
# Alias for convenience (add to ~/.zshrc or ~/.bashrc)
alias ml-run="docker compose exec ml-worker python /app/scripts/ml_pipeline.py"
alias ml-classify="docker compose exec ml-worker python /app/scripts/ml_classification.py"
alias ml-forecast="docker compose exec ml-worker python /app/scripts/ml_forecasting.py"
alias ml-spark="docker compose exec ml-worker python /app/scripts/spark_regression.py"

# Then use:
ml-run        # Run complete pipeline
ml-classify   # Run classification only
ml-forecast   # Run forecasting only
ml-spark      # Run Spark regression only
```

---

**📝 Note**: All commands assume you're in the project directory:
```bash
cd /Users/sandilyachimalamarri/data-pipeline-project
```

