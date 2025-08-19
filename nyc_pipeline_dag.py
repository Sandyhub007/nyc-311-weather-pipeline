from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Set Python path to your scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from fetch_311_to_postgres import main as fetch_311
from fetch_weather_to_postgres import main as fetch_weather
from join_311_weather import main as join_data

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 8, 1),
}

with DAG('nyc_data_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

    t1 = PythonOperator(
        task_id='fetch_311',
        python_callable=fetch_311
    )

    t2 = PythonOperator(
        task_id='fetch_weather',
        python_callable=fetch_weather
    )

    t3 = PythonOperator(
        task_id='join_data',
        python_callable=join_data
    )

    t1 >> t2 >> t3

