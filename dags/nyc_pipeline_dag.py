# dags/nyc_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import os
import sys

# Add path to import your scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

# Import the main functions from your scripts
from fetch_311_to_postgres import main as fetch_311
from fetch_weather_to_postgres import main as fetch_weather
from join_311_weather import main as join_data

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 8, 1),
    'retries': 1,
}

with DAG(
    dag_id='nyc_data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['nyc', 'weather', '311', 'ml'],
) as dag:

    # Data ingestion tasks
    task_fetch_311 = PythonOperator(
        task_id='fetch_311_data',
        python_callable=fetch_311
    )

    task_fetch_weather = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather
    )

    task_join = PythonOperator(
        task_id='join_311_and_weather',
        python_callable=join_data
    )

    # ML tasks - execute directly in ml-worker container
    # Note: These tasks are optional; comment out if not needed
    # task_ml_classification = BashOperator(
    #     task_id='ml_classification',
    #     bash_command='docker exec ml-worker python /app/scripts/ml_classification.py',
    # )
    #
    # task_ml_forecasting = BashOperator(
    #     task_id='ml_forecasting',
    #     bash_command='docker exec ml-worker python /app/scripts/ml_forecasting.py',
    # )
    #
    # task_spark_regression = BashOperator(
    #     task_id='spark_regression',
    #     bash_command='docker exec ml-worker python /app/scripts/spark_regression.py',
    # )
    #
    # # ML pipeline orchestrator (runs all ML tasks sequentially)
    # task_ml_pipeline = BashOperator(
    #     task_id='ml_pipeline_orchestrator',
    #     bash_command='docker exec ml-worker python /app/scripts/ml_pipeline.py',
    # )

    # Task dependencies
    # Data ingestion pipeline only (ML tasks are commented out)
    task_fetch_311 >> task_fetch_weather >> task_join
    
    # To enable ML tasks, uncomment the ML task definitions above and update dependencies:
    # task_join >> task_ml_pipeline
    # OR run tasks in parallel:
    # task_join >> [task_ml_classification, task_ml_forecasting, task_spark_regression]

