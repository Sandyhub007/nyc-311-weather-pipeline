-- models/staging/stg_nyc_311_with_weather.sql

SELECT
    id,
    created_at,
    complaint_type,
    descriptor,
    borough,
    latitude,
    longitude,
    temperature_c,
    precip_mm
FROM public.nyc_311_with_weather

