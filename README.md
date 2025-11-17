# NYC 311 + Weather Data Pipeline with Machine Learning

A fully automated, production-ready data pipeline that integrates NYC 311 service requests with weather data and provides machine learning capabilities for predictive analytics.

## 🚀 Project Overview

This project combines real-time data ingestion, data transformation, and machine learning to analyze NYC 311 complaints (specifically illegal parking in Bronx) and their correlation with weather patterns over the last 6 months.

### Author
- **Name:** Sandilya Chimalamarri
- **GitHub:** [@Sandyhub007](https://github.com/Sandyhub007)
- **Repository:** [nyc-311-weather-pipeline](https://github.com/Sandyhub007/nyc-311-weather-pipeline)

### License
MIT

---

## 📊 Tech Stack

### **Data Ingestion**
- Python, Requests, Pandas
- NYC 311 Socrata API
- Open-Meteo Weather API

### **Data Storage**
- PostgreSQL (Dockerized)
- Tables: `nyc_311`, `nyc_weather`, `nyc_311_with_weather`

### **Orchestration**
- Apache Airflow 2.8.1 (Dockerized)
- CeleryExecutor with Redis backend

### **Machine Learning**
- **Classification:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Time Series Forecasting:** Prophet
- **Regression:** Apache Spark MLlib (Linear Regression)
- **Libraries:** pandas, numpy, matplotlib, seaborn, joblib

### **Visualization**
- Metabase (Dockerized)
- Interactive dashboards and analytics

### **Infrastructure**
- Docker & Docker Compose
- Multi-container architecture with 7+ services

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                             │
│  • NYC 311 API (Illegal Parking - Bronx, Last 6 Months)        │
│  • Open-Meteo Weather API (Hourly Weather Data)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Apache Airflow DAG                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ fetch_311    │─▶│ fetch_weather│─▶│ join_data    │         │
│  │    data      │  │    data      │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PostgreSQL                                  │
│  • nyc_311: 42,881 records (Bronx illegal parking)             │
│  • nyc_weather: 4,344 hourly weather records                    │
│  • nyc_311_with_weather: Joined dataset                         │
└────────────────────────┬───────────────┬────────────────────────┘
                         │               │
          ┌──────────────┘               └──────────────┐
          ▼                                             ▼
┌─────────────────────┐                    ┌────────────────────────┐
│   Metabase          │                    │   ML Worker Container   │
│  • Dashboards       │                    │  ┌──────────────────┐  │
│  • Analytics        │                    │  │ Classification   │  │
│  • Visualizations   │                    │  │ (84% accuracy)   │  │
│  localhost:3000     │                    │  └──────────────────┘  │
└─────────────────────┘                    │  ┌──────────────────┐  │
                                           │  │ Forecasting      │  │
                                           │  │ (Prophet MAPE    │  │
                                           │  │  21.31%)         │  │
                                           │  └──────────────────┘  │
                                           │  ┌──────────────────┐  │
                                           │  │ Spark Regression │  │
                                           │  │ (R² 0.28)        │  │
                                           │  └──────────────────┘  │
                                           └────────────────────────┘
```

---

## 📋 Pipeline Steps

### **1. Data Ingestion**

#### `fetch_311_data`
- Fetches last 6 months of **illegal parking complaints** from **Bronx**
- API: NYC 311 Socrata API
- Output: `nyc_311` table (~42,881 records)
- Fields: `id`, `created_at`, `complaint_type`, `descriptor`, `borough`, `latitude`, `longitude`, `incident_address`, `city`

#### `fetch_weather_data`
- Fetches last 6 months of **hourly weather data** for NYC
- API: Open-Meteo Weather API
- Output: `nyc_weather` table (~4,344 records)
- Fields: `timestamp`, `temperature_c`, `precip_mm`, `humidity`, `wind_speed_kmh`, `weather_code`

#### `join_311_and_weather`
- Joins complaint data with weather data by timestamp
- Output: `nyc_311_with_weather` table
- Joins on hour-level alignment

### **2. Machine Learning Tasks**

#### `ml_classification.py` - Multiclass Complaint Type Prediction
- **Models:**
  - Logistic Regression (Baseline)
  - Random Forest
  - XGBoost (Best: **84% accuracy**)
- **Features:** Time-based (hour, day, month), location, weather (temperature, precipitation)
- **Output:** Trained models saved to `models/`, classification reports to `reports/`

#### `ml_forecasting.py` - Time Series Forecasting
- **Model:** Facebook Prophet
- **Predicts:** Total complaint volume for next 30 days
- **Metrics:** MAPE ~21.31%, RMSE, MAE
- **Output:** Forecast CSV and visualization plots

#### `spark_regression.py` - Hourly Complaint Volume Prediction
- **Model:** Spark MLlib Linear Regression (local mode)
- **Predicts:** Hourly complaint counts
- **Features:** Hour, day of week, month, temperature, precipitation
- **Metrics:** R² ~0.28, RMSE, MAE
- **Output:** Model saved in Spark format, sample predictions

---

## 🛠️ Setup Instructions

### **Prerequisites**
- Docker & Docker Compose installed
- Python 3.8+
- Git

### **1. Clone the Repository**
```bash
git clone https://github.com/Sandyhub007/nyc-311-weather-pipeline.git
cd nyc-311-weather-pipeline
```

### **2. Start All Services**
```bash
docker compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- Airflow Webserver (port 8080)
- Airflow Scheduler
- Airflow Worker
- Airflow Triggerer
- Metabase (port 3000)
- ML Worker (for ML tasks)

### **3. Verify Services**
```bash
docker compose ps
```

All services should show `Up` and `healthy`.

### **4. Access Airflow**
- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### **5. Access Metabase**
- URL: http://localhost:3000
- Email: `admin@metabase.local`
- Password: `metabase123`

---

## 🚦 Running the Pipeline

### **Trigger Airflow DAG**
```bash
# Unpause the DAG
docker compose exec airflow-scheduler airflow dags unpause nyc_data_pipeline

# Trigger a manual run
docker compose exec airflow-scheduler airflow dags trigger nyc_data_pipeline
```

### **Run ML Tasks (Manual)**
```bash
# Run all ML tasks
docker compose exec ml-worker python /app/scripts/ml_pipeline.py

# Or run individual tasks
docker compose exec ml-worker python /app/scripts/ml_classification.py
docker compose exec ml-worker python /app/scripts/ml_forecasting.py
docker compose exec ml-worker python /app/scripts/spark_regression.py
```

### **View Results**
- **Models:** Saved in `models/` directory
- **Reports:** Saved in `reports/` directory
- **Logs:** Available in Airflow UI task logs

---

## 📊 Machine Learning Results

### **Classification Performance (XGBoost)**
- **Accuracy:** 84%
- **Best Model:** XGBoost with hyperparameter tuning
- **Use Case:** Predict complaint type based on time, location, and weather

### **Time Series Forecasting (Prophet)**
- **MAPE:** 21.31%
- **Forecast Horizon:** 30 days
- **Use Case:** Predict future complaint volume trends

### **Regression (Spark MLlib)**
- **R² Score:** 0.28
- **Use Case:** Predict hourly complaint counts based on weather and temporal features

---

## 📈 Metabase Dashboards

### **Available Tables**
1. **nyc_311** - Raw 311 complaint data
2. **nyc_weather** - Hourly weather data
3. **nyc_311_with_weather** - Joined dataset (recommended for analysis)

### **Example Visualizations**
- Complaints over time (daily/hourly trends)
- Complaints vs temperature correlation
- Complaints on rainy days vs dry days
- Top complaint descriptors in Bronx
- Heatmap: complaints by hour and day of week

---

## 📂 Project Structure

```
nyc-311-weather-pipeline/
├── dags/
│   └── nyc_pipeline_dag.py          # Airflow DAG definition
├── scripts/
│   ├── fetch_311_to_postgres.py     # 311 data ingestion
│   ├── fetch_weather_to_postgres.py # Weather data ingestion
│   ├── join_311_weather.py          # Data joining
│   ├── ml_classification.py         # ML classification models
│   ├── ml_forecasting.py            # Time series forecasting
│   ├── spark_regression.py          # Spark regression
│   └── ml_pipeline.py               # ML orchestrator
├── models/                          # Saved ML models
├── reports/                         # ML reports and metrics
├── docker-compose.yaml              # Multi-container setup
├── Dockerfile.ml                    # ML worker container
├── requirements.txt                 # Python dependencies
├── ML_SETUP_GUIDE.md               # Detailed ML setup guide
├── ML_IMPLEMENTATION_SUMMARY.md    # Technical ML details
├── COMMANDS_QUICK_REFERENCE.md     # Command reference
├── DEPLOYMENT_CHECKLIST.md         # Deployment guide
├── README_ML.md                    # ML-specific README
├── validate_ml_setup.sh            # Setup validation script
└── README.md                       # This file
```

---

## 🔧 Configuration

### **Data Timeframe**
- **Current:** Last 6 months of data
- **Configurable in:** All `scripts/fetch_*.py` files

### **Database Connection**
- **Host:** `postgres` (within Docker network)
- **Port:** 5432
- **Database:** `airflow`
- **Username:** `airflow`
- **Password:** `airflow`

### **APIs**
- **NYC 311:** https://data.cityofnewyork.us/resource/erm2-nwe9.json
- **Weather:** https://open-meteo.com/en/docs

---

## 🧪 Testing & Validation

### **Validate ML Setup**
```bash
chmod +x validate_ml_setup.sh
./validate_ml_setup.sh
```

### **Test DAG Tasks**
```bash
# Test individual tasks
docker compose exec airflow-scheduler airflow tasks test nyc_data_pipeline fetch_311_data 2025-11-17
docker compose exec airflow-scheduler airflow tasks test nyc_data_pipeline fetch_weather_data 2025-11-17
docker compose exec airflow-scheduler airflow tasks test nyc_data_pipeline join_311_and_weather 2025-11-17
```

---

## 🐛 Troubleshooting

### **Airflow Web UI Not Loading**
```bash
# Check if another process is using port 8080
lsof -nP -iTCP:8080

# Restart Airflow webserver
docker compose restart airflow-webserver
```

### **ML Tasks Failing**
```bash
# Check ML worker logs
docker compose logs ml-worker

# Verify ML dependencies
docker compose exec ml-worker python -c "import sklearn, xgboost, prophet, pyspark; print('All ML libraries imported successfully')"
```

### **Database Connection Issues**
```bash
# Check PostgreSQL status
docker compose exec postgres psql -U airflow -d airflow -c "SELECT version();"

# Verify tables exist
docker compose exec postgres psql -U airflow -d airflow -c "\dt"
```

---

## 📚 Documentation

- **[ML Setup Guide](ML_SETUP_GUIDE.md)** - Comprehensive ML setup instructions
- **[ML Implementation Summary](ML_IMPLEMENTATION_SUMMARY.md)** - Technical ML details
- **[Commands Quick Reference](COMMANDS_QUICK_REFERENCE.md)** - All useful commands
- **[Deployment Checklist](DEPLOYMENT_CHECKLIST.md)** - Production deployment guide
- **[ML README](README_ML.md)** - ML-specific documentation

---

## 🎯 Key Features

✅ **Automated Data Ingestion** - Daily scheduled runs via Airflow  
✅ **Real-time Weather Integration** - Hourly weather data correlation  
✅ **Machine Learning Models** - Classification, forecasting, regression  
✅ **Interactive Dashboards** - Metabase for visual analytics  
✅ **Scalable Architecture** - Docker-based multi-container setup  
✅ **Production-Ready** - Health checks, retries, logging  
✅ **Comprehensive Documentation** - Setup guides and references  
✅ **Version Controlled** - Full Git history with detailed commits

---

## 🚀 Future Enhancements

- [ ] Add data quality tests (Great Expectations)
- [ ] Implement dbt models for advanced transformations
- [ ] Add deep learning models (LSTM for time series)
- [ ] Create API endpoints for model predictions
- [ ] Add CI/CD pipeline for automated testing
- [ ] Implement data versioning (DVC)
- [ ] Add more NYC boroughs and complaint types
- [ ] Real-time streaming with Kafka/Spark Streaming

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 Status

| Component          | Status |
|--------------------|--------|
| Data Ingestion     | ✅ Working |
| Airflow DAG        | ✅ Working |
| PostgreSQL         | ✅ Working |
| Metabase           | ✅ Working |
| ML Classification  | ✅ Working (84% accuracy) |
| ML Forecasting     | ✅ Working (21% MAPE) |
| ML Regression      | ✅ Working (R² 0.28) |
| Documentation      | ✅ Complete |
| dbt Models         | ⏳ Planned |
| Data Tests         | ⏳ Planned |
| API Export         | ⏳ Planned |

---

## 📞 Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/Sandyhub007/nyc-311-weather-pipeline/issues).

---

**Built with ❤️ using Python, Airflow, PostgreSQL, and Machine Learning**
