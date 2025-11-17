# ✅ ML Pipeline Deployment Checklist

## Pre-Deployment Verification

### 1. Files Created ✅
- [x] `docker-compose.yaml` - Updated with Spark and ML worker
- [x] `Dockerfile.ml` - ML worker container definition
- [x] `requirements.txt` - Python ML dependencies
- [x] `scripts/ml_classification.py` - Classification models
- [x] `scripts/ml_forecasting.py` - Prophet forecasting
- [x] `scripts/spark_regression.py` - Spark MLlib regression
- [x] `scripts/ml_pipeline.py` - Orchestrator
- [x] `dags/nyc_pipeline_dag.py` - Updated with ML tasks
- [x] `models/` directory - For trained models
- [x] `reports/` directory - For visualizations
- [x] Documentation files

## Deployment Steps

### Step 1: Start Docker Desktop ⏳
```bash
# Open Docker Desktop application
# Wait for "Docker Desktop is running" status
```
- [ ] Docker Desktop is running

### Step 2: Build ML Worker Image ⏳
```bash
cd /Users/sandilyachimalamarri/data-pipeline-project
docker compose build ml-worker
```
**Expected**: Building process completes successfully (~5-10 minutes first time)
- [ ] ML worker image built successfully
- [ ] No build errors in output

### Step 3: Start All Services ⏳
```bash
docker compose up -d
```
**Expected**: All containers start and become healthy
- [ ] PostgreSQL running
- [ ] Redis running
- [ ] Airflow services running (webserver, scheduler, worker)
- [ ] Metabase running
- [ ] **ML Worker running** ✨

### Step 4: Verify Services ⏳
```bash
docker compose ps
```
**Expected**: All services show "healthy" or "running" status
- [ ] All containers are up
- [ ] No containers in "restarting" or "unhealthy" state

### Step 5: Test ML Environment ⏳
```bash
# Test Python libraries
docker compose exec ml-worker python -c "import sklearn, xgboost, prophet, pyspark; print('✅ All ML libraries working!')"
```
**Expected**: `✅ All ML libraries working!`
- [ ] All ML libraries import successfully
- [ ] No ModuleNotFoundError

### Step 6: Verify Data Availability ⏳
```bash
# Check data
docker compose exec postgres psql -U airflow -d airflow -c "SELECT COUNT(*) FROM nyc_311_bronx_full_year;"
```
**Expected**: Returns count > 1,000,000 (from your full-year data fetch)
- [ ] Data table exists
- [ ] Sufficient data for ML training (>10,000 records minimum)

### Step 7: Run Classification Models ⏳
```bash
docker compose exec ml-worker python /app/scripts/ml_classification.py
```
**Expected Output**:
```
🤖 NYC 311 COMPLAINT TYPE CLASSIFICATION PIPELINE
🔍 Loading 50000 records from database...
✅ Loaded 50000 records
...
1️⃣  Training Logistic Regression...
   ✅ Logistic Regression trained
2️⃣  Training Random Forest...
   ✅ Random Forest trained
3️⃣  Training XGBoost...
   ✅ XGBoost trained
...
🎉 CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!
```
- [ ] Classification models trained
- [ ] Accuracy > 60%
- [ ] Models saved to `/app/models/`

### Step 8: Run Time Series Forecasting ⏳
```bash
docker compose exec ml-worker python /app/scripts/ml_forecasting.py
```
**Expected Output**:
```
🔮 NYC 311 TIME SERIES FORECASTING PIPELINE (PROPHET)
🔍 Loading time series data...
✅ Loaded XXX days of data
...
🤖 Training Prophet model...
   ✅ Model trained successfully
🔮 Generating 30-day forecast...
   ✅ Forecast generated
...
🎉 FORECASTING PIPELINE COMPLETED SUCCESSFULLY!
```
- [ ] Prophet model trained
- [ ] Forecast generated
- [ ] Plots saved to `/app/reports/`
- [ ] MAPE < 30%

### Step 9: Run Spark Regression ⏳
```bash
docker compose exec ml-worker python /app/scripts/spark_regression.py
```
**Expected Output**:
```
⚡ NYC 311 SPARK MLlib LINEAR REGRESSION PIPELINE
🚀 Creating Spark session...
✅ Spark session created
...
🤖 Training Spark MLlib Linear Regression...
   ✅ Model trained
📊 MODEL PERFORMANCE:
   Root Mean Squared Error (RMSE): XX.XX
   R² Score: 0.XXXX
...
🎉 SPARK MLlib PIPELINE COMPLETED SUCCESSFULLY!
```
- [ ] Spark session created
- [ ] Model trained successfully
- [ ] R² score > 0.30

### Step 10: Run Complete Pipeline ⏳
```bash
docker compose exec ml-worker python /app/scripts/ml_pipeline.py
```
**Expected**: All 3 ML tasks complete sequentially
- [ ] Classification completes
- [ ] Forecasting completes
- [ ] Spark regression completes
- [ ] Exit code 0 (success)

### Step 11: Verify Airflow Integration ⏳
```bash
# Access Airflow UI
open http://localhost:8080
# Login: airflow / airflow
```
- [ ] Airflow UI accessible
- [ ] `nyc_data_pipeline` DAG visible
- [ ] DAG shows new ML tasks
- [ ] Can trigger DAG manually

### Step 12: Trigger Full DAG Run ⏳
In Airflow UI:
1. Find `nyc_data_pipeline`
2. Unpause the DAG (toggle to green)
3. Click "Play" button to trigger
4. Watch tasks execute

**Expected Flow**:
```
fetch_311_data → fetch_weather_data → join_311_and_weather → ml_pipeline_orchestrator
```
- [ ] Data tasks complete
- [ ] ML pipeline task starts
- [ ] All tasks show green (success)

### Step 13: Check Generated Outputs ⏳
```bash
# Check models
ls -lh models/

# Check reports
ls -lh reports/

# Check database
docker compose exec postgres psql -U airflow -d airflow -c "SELECT * FROM ml_forecast_results LIMIT 5;"
```
- [ ] Model files exist (.pkl files)
- [ ] Report files exist (.png files)
- [ ] Forecast table populated

### Step 14: Access Metabase ⏳
```bash
open http://localhost:3000
```
- [ ] Metabase accessible
- [ ] Can connect to PostgreSQL
- [ ] Can query `ml_forecast_results` table

## Post-Deployment Verification

### Smoke Tests ✅
```bash
# Quick health check script
docker compose ps | grep -E "(healthy|running)" | wc -l
# Should show 7+ services

docker compose exec ml-worker python -c "import os; print('Models:', len([f for f in os.listdir('/app/models') if f.endswith('.pkl')]))"
# Should show multiple model files

docker compose exec postgres psql -U airflow -d airflow -c "SELECT COUNT(*) FROM ml_forecast_results;"
# Should return > 0 rows
```
- [ ] All smoke tests pass

### Performance Check ✅
```bash
# Check resource usage
docker stats --no-stream ml-worker
```
- [ ] Memory usage < 4GB
- [ ] CPU usage reasonable
- [ ] No container restarts

## Troubleshooting Guide

### Issue: Build fails with dependency errors
**Solution**:
```bash
docker compose build --no-cache ml-worker
```

### Issue: Containers won't start
**Solution**:
```bash
docker compose down
docker compose up -d
docker compose logs ml-worker
```

### Issue: Python import errors
**Solution**:
```bash
docker compose exec ml-worker pip install --upgrade scikit-learn xgboost prophet pyspark
```

### Issue: Database connection fails
**Solution**:
```bash
docker compose restart postgres
# Wait 30 seconds
docker compose exec postgres psql -U airflow -d airflow -c "SELECT 1"
```

## Success Criteria 🎯

Your deployment is successful when ALL of these are true:
- ✅ All Docker containers running healthy
- ✅ ML classification accuracy > 60%
- ✅ Prophet forecast MAPE < 30%
- ✅ Spark regression R² > 0.30
- ✅ Models saved in `/models/` directory
- ✅ Forecast plots in `/reports/` directory
- ✅ Airflow DAG runs end-to-end successfully
- ✅ Metabase can query forecast results

## Final Validation Command
```bash
# Run this to validate everything
cd /Users/sandilyachimalamarri/data-pipeline-project
./validate_ml_setup.sh
```

## Need Help?

1. **Check logs**: `docker compose logs ml-worker --tail=50`
2. **Review docs**: `ML_SETUP_GUIDE.md`
3. **Quick ref**: `COMMANDS_QUICK_REFERENCE.md`
4. **Summary**: `ML_IMPLEMENTATION_SUMMARY.md`

---

## Sign-off

When all items are checked:
- Deployment Date: _______________
- Deployed By: _______________
- All Tests Passing: [ ] Yes / [ ] No
- Production Ready: [ ] Yes / [ ] No

**🎉 Congratulations! Your ML pipeline is now live!** 🚀

