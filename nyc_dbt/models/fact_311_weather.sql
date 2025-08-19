-- models/fact_311_weather.sql

SELECT
    id,
    created_at,
    complaint_type,
    borough,
    ROUND(CAST(temperature_c AS NUMERIC), 1) AS temp_c,
    ROUND(CAST(precip_mm AS NUMERIC), 1) AS rain_mm
FROM {{ ref('stg_nyc_311_with_weather') }}

