# 🤖 NYC 311 ML Pipeline Setup Guide

## 📋 Overview

This ML pipeline adds predictive capabilities to your NYC 311 data pipeline with:
- **Multiclass Classification**: Predict complaint types using Logistic Regression, Random Forest, and XGBoost
- **Time Series Forecasting**: Predict daily complaint volumes using Facebook Prophet
- **Spark MLlib Regression**: Scalable hourly complaint volume predictions

## 🚀 Quick Start

### Step 1: Build the ML Worker Image

```bash
cd /Users/sandilyachimalamarri/data-pipeline-project
docker compose build ml-worker
```

### Step 2: Start All Services

```bash
docker compose up -d
```

This will start:
- PostgreSQL (database)
- Redis (message broker)
- Airflow (webserver, scheduler, worker)
- Metabase (visualization)
- **ML Worker** (ML pipeline executor)

> ℹ️ Spark MLlib runs in local mode inside the `ml-worker` container using PySpark, so no separate Spark service is required.

### Step 3: Verify Services are Running

```bash
docker compose ps
```

You should see all services as "healthy" or "running".

## 🧪 Testing the ML Pipeline

### Option 1: Run Complete ML Pipeline

Run all ML tasks sequentially:

```bash
docker compose exec ml-worker python /app/scripts/ml_pipeline.py
```

### Option 2: Run Individual ML Tasks

#### Classification (Logistic Regression, Random Forest, XGBoost):
```bash
docker compose exec ml-worker python /app/scripts/ml_classification.py
```

Expected output:
- Training on 50,000 samples
- 60-80% accuracy on complaint type prediction
- Models saved to `models/` directory

#### Time Series Forecasting (Prophet):
```bash
docker compose exec ml-worker python /app/scripts/ml_forecasting.py
```

Expected output:
- 30-day forecast
- Forecast plots saved to `reports/` directory
- MAPE typically < 20%

#### Spark MLlib Linear Regression:
```bash
docker compose exec ml-worker python /app/scripts/spark_regression.py
```

Expected output:
- Hourly complaint volume predictions
- R² score and RMSE metrics
- Models saved to `models/` directory

### Option 3: Run via Airflow DAG

1. Go to Airflow UI: http://localhost:8080
2. Find the `nyc_data_pipeline` DAG
3. Unpause the DAG
4. Trigger a manual run
5. The ML pipeline will run automatically after data ingestion

## 📊 Accessing Results

### Models
All trained models are saved in:
```
/Users/sandilyachimalamarri/data-pipeline-project/models/
```

Files:
- `logistic_model.pkl` - Logistic Regression
- `random_forest_model.pkl` - Random Forest  
- `xgboost_model.pkl` - XGBoost
- `prophet_forecaster.pkl` - Prophet time series model
- `spark_lr_model/` - Spark MLlib model
- `*_encoder.pkl` - Label encoders for categorical features

### Reports & Visualizations
Forecast plots and analysis reports:
```
/Users/sandilyachimalamarri/data-pipeline-project/reports/
```

Files:
- `forecast_plot.png` - 30-day complaint volume forecast
- `forecast_components.png` - Trend, weekly, and seasonal components

### Database Tables
ML results are stored in PostgreSQL:

```sql
-- Time series forecasts
SELECT * FROM ml_forecast_results LIMIT 10;

-- View actual vs predicted
SELECT 
    date,
    predicted_complaints,
    lower_bound,
    upper_bound
FROM ml_forecast_results
WHERE date > CURRENT_DATE
ORDER BY date;
```

## 🎯 Metabase Visualizations

### Create ML Dashboards

1. Go to Metabase: http://localhost:3000
2. Connect to database (if not already): `postgres:5432/airflow`
3. Create questions using ML tables:

#### Forecast Visualization Query:
```sql
SELECT 
    date,
    predicted_complaints,
    lower_bound,
    upper_bound
FROM ml_forecast_results
ORDER BY date
```

Chart type: Line chart with confidence bands

#### Model Performance Query:
```sql
SELECT 
    complaint_type,
    COUNT(*) as actual_count
FROM nyc_311_bronx_full_year
GROUP BY complaint_type
ORDER BY actual_count DESC
LIMIT 10
```

## 🔧 Troubleshooting

### Issue: "Docker daemon not running"
**Solution**: Start Docker Desktop

### Issue: "ml-worker container not found"
**Solution**: Build and start services:
```bash
docker compose build ml-worker
docker compose up -d ml-worker
```

### Issue: "ModuleNotFoundError" in ML scripts
**Solution**: Rebuild the ML worker image:
```bash
docker compose build --no-cache ml-worker
docker compose restart ml-worker
```

### Issue: "Connection to Postgres failed"
**Solution**: Verify database is running:
```bash
docker compose ps postgres
docker compose exec postgres psql -U airflow -d airflow -c "SELECT 1"
```

### Issue: "Out of memory" errors
**Solution**: Reduce sample size in classification:
- Edit `scripts/ml_classification.py`
- Change `sample_size=50000` to `sample_size=10000`

## 📈 Performance Expectations

### Classification Models

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| Logistic Regression | 60-70% | ~30 seconds |
| Random Forest | 70-80% | ~2 minutes |
| XGBoost | 75-85% | ~3 minutes |

### Time Series Forecasting

| Metric | Expected Value |
|--------|---------------|
| MAPE | 15-25% |
| MAE | 50-150 complaints/day |
| Training Time | ~1-2 minutes |

### Spark Regression

| Metric | Expected Value |
|--------|---------------|
| R² Score | 0.40-0.60 |
| RMSE | 20-40 complaints/hour |
| Training Time | ~1 minute |

## 🔄 Retraining Models

Models should be retrained periodically as new data arrives:

### Manual Retraining
```bash
# Retrain all models
docker compose exec ml-worker python /app/scripts/ml_pipeline.py

# Or retrain specific models
docker compose exec ml-worker python /app/scripts/ml_classification.py
docker compose exec ml-worker python /app/scripts/ml_forecasting.py
```

### Automated Retraining
The Airflow DAG is configured to run daily (@daily schedule), so models will be automatically retrained with fresh data each day.

## 📚 Understanding the Results

### Classification Output
The classification models predict which complaint type is most likely based on:
- **Time features**: hour, day of week, month, weekend indicator
- **Location features**: latitude, longitude, borough
- **Context features**: descriptor

**Use case**: Predict what type of complaint will arrive next based on current conditions.

### Forecasting Output
Prophet generates:
- **Point forecast**: Most likely complaint volume
- **Confidence interval**: 95% probability range
- **Components**: Trend, weekly patterns, seasonal patterns

**Use case**: Plan resource allocation for next 30 days based on predicted complaint volumes.

### Spark Regression Output
MLlib predicts hourly complaint volumes based on:
- Hour of day
- Day of week
- Weekend indicator
- Business hours indicator

**Use case**: Real-time prediction of complaint volumes for current hour.

## 🎓 Next Steps

1. **Fine-tune hyperparameters** in each ML script
2. **Add more features** (e.g., weather data, holidays)
3. **Create custom visualizations** in Metabase
4. **Set up model monitoring** to track performance over time
5. **Add alerting** for anomalous predictions

## 📞 Support

For issues or questions:
1. Check Docker logs: `docker compose logs ml-worker`
2. Verify data availability: Check `nyc_311_bronx_full_year` table has data
3. Review ML script outputs in console

---

**🎉 Congratulations!** You now have a complete ML-powered NYC 311 data pipeline!

