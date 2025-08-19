# dags/nyc_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
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
    tags=['nyc', 'weather', '311'],
) as dag:

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

    task_fetch_311 >> task_fetch_weather >> task_join

