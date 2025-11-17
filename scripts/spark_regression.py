#!/usr/bin/env python3
"""
Spark MLlib Linear Regression for NYC 311 Complaints
Scalable hourly complaint volume prediction
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, hour, dayofweek, month, count, avg
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
from datetime import datetime
import pandas as pd

def create_spark_session():
    """Create Spark session"""
    print("🚀 Creating Spark session...")
    
    spark = SparkSession.builder \
        .appName("NYC311_Regression") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("✅ Spark session created")
    
    return spark

def load_data_from_postgres(spark):
    """Load data from PostgreSQL"""
    print("\n🔍 Loading data from PostgreSQL...")
    
    jdbc_url = "jdbc:postgresql://postgres:5432/airflow"
    properties = {
        "user": "airflow",
        "password": "airflow",
        "driver": "org.postgresql.Driver"
    }
    
    # Load Bronx data
    query = "(SELECT * FROM nyc_311_bronx_full_year WHERE created_at >= NOW() - INTERVAL '180 days') AS six_months"
    df = spark.read.jdbc(
        url=jdbc_url,
        table=query,
        properties=properties
    )
    
    record_count = df.count()
    print(f"✅ Loaded {record_count:,} records from PostgreSQL")
    
    return df

def prepare_features(df):
    """Prepare features for regression"""
    print("\n🔧 Preparing features...")
    
    # Extract temporal features
    df = df.withColumn("hour", hour("created_at"))
    df = df.withColumn("day_of_week", dayofweek("created_at"))
    df = df.withColumn("month", month("created_at"))
    df = df.withColumn("date", date_format("created_at", "yyyy-MM-dd"))
    
    # Convert lat/lon to numeric
    df = df.withColumn("latitude", col("latitude").cast("double"))
    df = df.withColumn("longitude", col("longitude").cast("double"))
    
    # Group by date and hour for hourly complaint volume prediction
    hourly_data = df.groupBy("date", "hour").agg(
        count("*").alias("complaint_count")
    )
    
    # Add temporal features back
    hourly_data = hourly_data.withColumn("day_of_week", dayofweek(col("date")))
    hourly_data = hourly_data.withColumn("month", month(col("date")))
    
    # Add derived features
    hourly_data = hourly_data.withColumn(
        "is_weekend", 
        (col("day_of_week").isin([1, 7])).cast("int")
    )
    hourly_data = hourly_data.withColumn(
        "is_night", 
        ((col("hour") >= 22) | (col("hour") <= 6)).cast("int")
    )
    hourly_data = hourly_data.withColumn(
        "is_business_hours", 
        ((col("hour") >= 9) & (col("hour") <= 17)).cast("int")
    )
    
    # Remove rows with null values
    hourly_data = hourly_data.na.drop()
    
    feature_count = hourly_data.count()
    print(f"✅ Prepared {feature_count:,} hourly records")
    
    return hourly_data

def train_linear_regression(hourly_data):
    """Train Linear Regression model"""
    print("\n🤖 Training Spark MLlib Linear Regression...")
    
    # Select features for prediction
    feature_cols = [
        "hour", "day_of_week", "month", 
        "is_weekend", "is_night", "is_business_hours"
    ]
    
    print(f"   Features: {', '.join(feature_cols)}")
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(hourly_data)
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(data)
    scaled_data = scaler_model.transform(data)
    
    # Split data
    train_data, test_data = scaled_data.randomSplit([0.8, 0.2], seed=42)
    
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"   📊 Training set: {train_count:,} samples")
    print(f"   📊 Test set: {test_count:,} samples")
    
    # Train Linear Regression
    lr = LinearRegression(
        featuresCol="scaled_features",
        labelCol="complaint_count",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.1
    )
    
    print("   Training model...")
    lr_model = lr.fit(train_data)
    print("   ✅ Model trained")
    
    # Make predictions
    predictions = lr_model.transform(test_data)
    
    # Evaluate model
    evaluator_rmse = RegressionEvaluator(
        labelCol="complaint_count",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    evaluator_r2 = RegressionEvaluator(
        labelCol="complaint_count",
        predictionCol="prediction",
        metricName="r2"
    )
    
    evaluator_mae = RegressionEvaluator(
        labelCol="complaint_count",
        predictionCol="prediction",
        metricName="mae"
    )
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean Absolute Error (MAE): {mae:.2f}")
    
    # Print coefficients
    print(f"\n📈 MODEL COEFFICIENTS:")
    for i, col in enumerate(feature_cols):
        print(f"   {col}: {lr_model.coefficients[i]:.4f}")
    print(f"   Intercept: {lr_model.intercept:.4f}")
    
    return lr_model, scaler_model, predictions, {'rmse': rmse, 'r2': r2, 'mae': mae}

def save_spark_model(model, scaler_model, path="/app/models"):
    """Save Spark ML models"""
    print(f"\n💾 Saving Spark models to {path}/...")
    
    try:
        os.makedirs(path, exist_ok=True)
        model.write().overwrite().save(f"{path}/spark_lr_model")
        scaler_model.write().overwrite().save(f"{path}/spark_scaler")
        print("   ✅ Spark models saved successfully")
    except Exception as e:
        print(f"   ⚠️  Could not save models: {e}")

def show_sample_predictions(predictions, spark):
    """Show sample predictions"""
    print(f"\n📋 SAMPLE PREDICTIONS:")
    
    sample = predictions.select(
        "hour", "day_of_week", "complaint_count", "prediction"
    ).orderBy("hour").limit(24)
    
    sample_pandas = sample.toPandas()
    
    print("\n   Hour | Day of Week | Actual | Predicted")
    print("   " + "-" * 45)
    for _, row in sample_pandas.iterrows():
        hour = int(round(row['hour'])) if not pd.isna(row['hour']) else None
        dow = int(round(row['day_of_week'])) if not pd.isna(row['day_of_week']) else None
        actual = float(row['complaint_count']) if not pd.isna(row['complaint_count']) else None
        pred = float(row['prediction']) if not pd.isna(row['prediction']) else None

        hour_str = f"{hour:4d}" if hour is not None else "   -"
        dow_str = f"{dow:11d}" if dow is not None else "     -"
        actual_str = f"{actual:6.0f}" if actual is not None else "     -"
        pred_str = f"{pred:9.2f}" if pred is not None else "    -"

        print(f"   {hour_str} | {dow_str} | {actual_str} | {pred_str}")

def main():
    print("="*70)
    print("⚡ NYC 311 SPARK MLlib LINEAR REGRESSION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    
    spark = None
    
    try:
        # Create Spark session
        spark = create_spark_session()
        
        # Load data
        df = load_data_from_postgres(spark)
        
        # Prepare features
        prepared_data = prepare_features(df)
        
        # Train model
        model, scaler_model, predictions, metrics = train_linear_regression(prepared_data)
        
        # Show sample predictions
        show_sample_predictions(predictions, spark)
        
        # Save models
        save_spark_model(model, scaler_model)
        
        print(f"\n{'='*70}")
        print("🎉 SPARK MLlib PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Finished at: {datetime.now()}")
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   RMSE: {metrics['rmse']:.2f} complaints/hour")
        print(f"   R² Score: {metrics['r2']:.4f}")
        print(f"   MAE: {metrics['mae']:.2f} complaints/hour")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if spark:
            spark.stop()
            print("\n🛑 Spark session stopped")

if __name__ == "__main__":
    main()

