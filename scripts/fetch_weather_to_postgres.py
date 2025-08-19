# scripts/fetch_weather_to_postgres.py

def main():
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    from sqlalchemy import create_engine

    # Define NYC coordinates
    latitude = 40.7128
    longitude = -74.0060

    # Get yesterday's date (you can change this for testing)
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    start_date = end_date = yesterday.strftime('%Y-%m-%d')

    # Open-Meteo API (no key needed)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation",
        "timezone": "America/New_York"
    }

    print(f"Fetching weather for {start_date}...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather data: {response.status_code}")

    data = response.json()
    hours = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    precip = data["hourly"]["precipitation"]

    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hours),
        "temperature_c": temps,
        "precip_mm": precip
    })

    print("Fetched weather rows:", len(df))

    # Connect to Postgres and write to table
    engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/postgres")
    df.to_sql("nyc_weather", engine, if_exists="replace", index=False)

    print("✅ Done! Weather data saved to Postgres.")

# Only runs if you execute this file directly
if __name__ == "__main__":
    main()

