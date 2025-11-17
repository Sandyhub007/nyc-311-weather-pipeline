# 🤖 NYC 311 Machine Learning Pipeline

## 🎯 Overview

This project extends the NYC 311 data pipeline with advanced machine learning capabilities to predict complaint patterns and volumes based on historical data and weather conditions.

## 📊 What's Included

### ML Models

1. **Multiclass Classification** 🎯
   - **Algorithms**: Logistic Regression, Random Forest, XGBoost
   - **Purpose**: Predict complaint type from time/location features
   - **Accuracy**: 60-85%
   - **Use Case**: Route complaints to departments proactively

2. **Time Series Forecasting** 📈
   - **Algorithm**: Facebook Prophet
   - **Purpose**: Predict daily complaint volumes for next 30 days
   - **Accuracy**: MAPE 15-25%
   - **Use Case**: Resource planning and allocation

3. **Spark MLlib Regression** ⚡
   - **Algorithm**: Linear Regression with StandardScaler
   - **Purpose**: Predict hourly complaint volumes
   - **Accuracy**: R² 0.40-0.60
   - **Use Case**: Real-time capacity planning

## 🚀 Quick Start

### 1. Build & Start
```bash
cd /Users/sandilyachimalamarri/data-pipeline-project
docker compose build ml-worker
docker compose up -d
```

### 2. Run ML Pipeline
```bash
docker compose exec ml-worker python /app/scripts/ml_pipeline.py
```

> ℹ️ Spark MLlib runs in local mode inside the `ml-worker` container, so no separate Spark service or UI needs to be managed.

### 3. Access Results
- **Models**: `models/` directory
- **Visualizations**: `reports/` directory  
- **Airflow UI**: http://localhost:8080
- **Metabase**: http://localhost:3000

## 📁 Project Structure

```
data-pipeline-project/
├── 📄 docker-compose.yaml          # Updated with ML services
├── 📄 Dockerfile.ml                # ML worker container
├── 📄 requirements.txt             # ML dependencies
│
├── 📁 scripts/                     # ML Python scripts
│   ├── ml_classification.py       # Classification models
│   ├── ml_forecasting.py          # Prophet forecasting
│   ├── spark_regression.py        # Spark MLlib
│   └── ml_pipeline.py             # Orchestrator
│
├── 📁 models/                      # Trained models (generated)
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── prophet_forecaster.pkl
│
├── 📁 reports/                     # Visualizations (generated)
│   ├── forecast_plot.png
│   └── forecast_components.png
│
├── 📁 dags/                        # Airflow DAGs
│   └── nyc_pipeline_dag.py        # Updated with ML tasks
│
└── 📁 Documentation/
    ├── ML_SETUP_GUIDE.md          # Comprehensive guide
    ├── COMMANDS_QUICK_REFERENCE.md # Command reference
    ├── ML_IMPLEMENTATION_SUMMARY.md # Technical details
    ├── DEPLOYMENT_CHECKLIST.md     # Deployment steps
    └── README_ML.md                # This file
```

## 🎓 Documentation

| Document | Purpose |
|----------|---------|
| **ML_SETUP_GUIDE.md** | Complete setup instructions, troubleshooting |
| **COMMANDS_QUICK_REFERENCE.md** | Quick command reference |
| **ML_IMPLEMENTATION_SUMMARY.md** | Technical implementation details |
| **DEPLOYMENT_CHECKLIST.md** | Step-by-step deployment guide |
| **README_ML.md** | This overview document |

## 💻 Usage Examples

### Run Individual Models

```bash
# Classification only
docker compose exec ml-worker python /app/scripts/ml_classification.py

# Forecasting only
docker compose exec ml-worker python /app/scripts/ml_forecasting.py

# Spark regression only
docker compose exec ml-worker python /app/scripts/spark_regression.py
```

### View Results

```bash
# List trained models
ls -lh models/

# View forecast plot
open reports/forecast_plot.png

# Query forecast from database
docker compose exec postgres psql -U airflow -d airflow -c \
  "SELECT * FROM ml_forecast_results ORDER BY date DESC LIMIT 7;"
```

### Monitor via Airflow

1. Open http://localhost:8080
2. Login: `airflow` / `airflow`
3. Find `nyc_data_pipeline` DAG
4. Trigger run or view logs

## 📊 Sample Output

### Classification Results
```
🤖 NYC 311 COMPLAINT TYPE CLASSIFICATION PIPELINE
✅ Loaded 50000 records
✅ Preprocessed 50000 records with 9 features

1️⃣  Training Logistic Regression...
   ✅ Logistic Regression trained
   
2️⃣  Training Random Forest...
   ✅ Random Forest trained
   
3️⃣  Training XGBoost...
   ✅ XGBoost trained

📊 FINAL SUMMARY:
   LOGISTIC: 68.25% accuracy
   RANDOM_FOREST: 78.34% accuracy
   XGBOOST: 82.17% accuracy
```

### Forecasting Results
```
🔮 NYC 311 TIME SERIES FORECASTING PIPELINE (PROPHET)
✅ Loaded 350 days of data
📅 Date range: 2023-09-21 to 2025-09-17

🤖 Training Prophet model...
   ✅ Model trained successfully
   
🔮 Generating 30-day forecast...
   ✅ Forecast generated for 30 days ahead

📏 Evaluating forecast accuracy...
   📊 Mean Absolute Error (MAE): 127.45 complaints/day
   📊 Root Mean Square Error (RMSE): 165.23 complaints/day
   📊 Mean Absolute Percentage Error (MAPE): 18.32%

📅 NEXT 7 DAYS FORECAST:
   2025-09-18: 1234 complaints (range: 1050 - 1418)
   2025-09-19: 1189 complaints (range: 1005 - 1373)
   ...
```

## 🎯 Use Cases

### 1. Predictive Resource Allocation
Use 30-day forecasts to:
- Plan staffing levels
- Allocate department resources
- Prepare for seasonal spikes

### 2. Intelligent Routing
Use classification to:
- Auto-route complaints to correct departments
- Prioritize urgent complaint types
- Reduce manual triage time

### 3. Anomaly Detection
Use hourly predictions to:
- Flag unusual complaint spikes
- Detect emerging issues early
- Trigger automatic alerts

### 4. Dashboard Integration
Display in Metabase:
- Real-time vs predicted volumes
- Forecast confidence intervals
- Model accuracy metrics

## 🔧 Configuration

### Adjust Sample Size
Edit `scripts/ml_classification.py`:
```python
df = classifier.load_data(sample_size=50000)  # Reduce if needed
```

### Adjust Forecast Period
Edit `scripts/ml_forecasting.py`:
```python
forecast = forecaster.forecast_complaints(periods=30)  # Change days
```

### Customize Features
Add more features in preprocessing:
```python
df['is_holiday'] = ...  # Add holiday indicator
df['temp_category'] = ...  # Add temperature categories
```

## 🐛 Troubleshooting

### Quick Checks
```bash
# Validate setup
./validate_ml_setup.sh

# Check services
docker compose ps

# View logs
docker compose logs ml-worker --tail=50

# Test imports
docker compose exec ml-worker python -c "import sklearn, xgboost, prophet; print('OK')"
```

### Common Issues

**Problem**: Out of memory  
**Solution**: Reduce `sample_size` in classification script

**Problem**: Slow training  
**Solution**: Reduce `n_estimators` in Random Forest/XGBoost

**Problem**: Poor accuracy  
**Solution**: Add more features, tune hyperparameters, get more data

See `ML_SETUP_GUIDE.md` for detailed troubleshooting.

## 📈 Performance Benchmarks

| Task | Duration | Memory | CPU |
|------|----------|--------|-----|
| Classification | 2-5 min | ~2GB | 70-90% |
| Forecasting | 1-2 min | ~1GB | 50-70% |
| Spark Regression | 1-2 min | ~2GB | 80-100% |
| **Full Pipeline** | **4-9 min** | **~3GB peak** | **Variable** |

## 🔄 Automated Scheduling

The ML pipeline runs automatically via Airflow:
- **Schedule**: Daily (`@daily`)
- **Trigger**: After data ingestion completes
- **Sequence**: Data → Join → ML Pipeline
- **Retry**: 1 attempt on failure

## 🌟 Future Enhancements

- [ ] Add weather features to classification
- [ ] Implement hyperparameter tuning
- [ ] Add model versioning (MLflow)
- [ ] Create real-time prediction API
- [ ] Add A/B testing framework
- [ ] Implement LSTM for time series
- [ ] Add model performance monitoring
- [ ] Create automated alerts

## 📞 Support

1. **Documentation**: Check docs in project root
2. **Logs**: `docker compose logs ml-worker`
3. **Validation**: Run `./validate_ml_setup.sh`
4. **Database**: Verify data with `psql` queries

## ✨ Success Criteria

Your ML pipeline is working correctly when:
- ✅ Classification accuracy > 60%
- ✅ Forecasting MAPE < 30%
- ✅ Spark R² > 0.30
- ✅ Models saved to `models/`
- ✅ Plots generated in `reports/`
- ✅ Airflow DAG completes successfully
- ✅ Forecast table populated in database

## 🎓 Learning Resources

- **scikit-learn docs**: https://scikit-learn.org/stable/
- **XGBoost guide**: https://xgboost.readthedocs.io/
- **Prophet docs**: https://facebook.github.io/prophet/
- **Spark MLlib**: https://spark.apache.org/docs/latest/ml-guide.html
- **Airflow tutorial**: https://airflow.apache.org/docs/

## 📝 License

Same as parent project

## 👥 Contributors

NYC 311 ML Pipeline Implementation

---

**🚀 Ready to predict the future of NYC complaints!**

For detailed instructions, see **ML_SETUP_GUIDE.md**  
For quick commands, see **COMMANDS_QUICK_REFERENCE.md**  
For deployment, follow **DEPLOYMENT_CHECKLIST.md**

