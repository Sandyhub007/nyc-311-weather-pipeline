project:
  name: NYC 311 + Weather Data Pipeline
  description: >
    A fully automated, open-source data pipeline that pulls and joins NYC 311 service requests and historical weather data using Python, Airflow, dbt, and Metabase.
  author:
    name: Your Name
    github: yourusername
  license: MIT
  stack:
    ingestion: [Python, Requests, Pandas]
    transformation: [Pandas, dbt]
    storage: PostgreSQL (Dockerized)
    orchestration: Apache Airflow (Dockerized)
    visualization: Metabase (Dockerized)
    infrastructure: [Docker, Docker Compose]
  tools:
    - Python
    - PostgreSQL
    - Apache Airflow
    - dbt
    - Metabase
    - Docker

pipeline:
  steps:
    - name: fetch_311_data
      type: Python
      description: >
        Pulls fresh NYC 311 service request data from the Socrata API and loads it into the `nyc_311` Postgres table.
    - name: fetch_weather_data
      type: Python
      description: >
        Pulls hourly historical weather data for NYC using the Open-Meteo API and loads it into the `nyc_weather` Postgres table.
    - name: join_311_and_weather
      type: Python
      description: >
        Joins complaint data with hourly weather readings by timestamp and stores the output in `nyc_311_with_weather`.
    - name: dbt_run
      type: dbt
      models:
        - stg_nyc_311_with_weather
        - fact_311_weather

models:
  - name: stg_nyc_311_with_weather
    description: >
      Staging model that selects and formats joined complaint + weather data.
    materialization: view
  - name: fact_311_weather
    description: >
      Final curated model with rounded temperature and precipitation data for analytics and dashboards.
    materialization: view

dashboards:
  platform: Metabase
  charts:
    - name: Rat complaints vs rainfall
      query: Group by day, filter complaint_type = 'rat', compare count vs avg rain_mm
    - name: Complaints vs temperature
      query: Group by day, compare count vs avg temp_c
    - name: Top complaints on rainy days
      query: Filter rain_mm > 0, group by complaint_type

instructions:
  setup:
    - Install Python and Docker
    - Create a virtualenv and install dependencies
    - Run all services using docker-compose
    - Log in to Airflow (localhost:8080) and Metabase (localhost:3000)
  usage:
    - Trigger `nyc_data_pipeline` DAG in Airflow
    - Run `dbt run` to materialize models
    - Explore models in Metabase
  dev_commands:
    - dbt debug
    - dbt run
    - dbt clean
    - dbt docs generate

data_sources:
  - name: NYC 311 Complaints
    url: https://data.cityofnewyork.us/resource/erm2-nwe9.json
  - name: Open-Meteo Weather API
    url: https://open-meteo.com

status:
  ingestion: true
  airflow: true
  dbt_models: true
  dashboards: true
  dbt_tests: false
  data_validation: false
  api_export: false

