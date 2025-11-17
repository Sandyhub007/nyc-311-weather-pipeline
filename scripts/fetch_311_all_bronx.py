# scripts/fetch_311_all_bronx.py
# Alternative script to get ALL complaint types from BRONX for comparison

def main():
    import requests
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta

    # NYC 311 API endpoint
    API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    
    # Calculate date range for last 1 year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Parameters for ALL BRONX complaints (not just parking)
    params = {
        "$limit": 50000,  # Maximum allowed by NYC Open Data API
        "$order": "created_date DESC",
        "$where": f"created_date >= '{start_date.strftime('%Y-%m-%d')}' AND created_date <= '{end_date.strftime('%Y-%m-%d')}' AND borough = 'BRONX'",
        "$select": "unique_key,created_date,complaint_type,descriptor,borough,latitude,longitude,incident_address,city"
    }

    print(f"🔍 Fetching 1 year of ALL complaints from BRONX...")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data)

    print(f"📊 Fetched {len(df)} total complaints from BRONX")

    if len(df) == 0:
        print("⚠️  No data found for the specified criteria")
        return

    # Select and rename important columns
    available_columns = df.columns.tolist()
    required_columns = ["unique_key", "created_date", "complaint_type", "descriptor", "borough"]
    optional_columns = ["latitude", "longitude", "incident_address", "city"]
    
    columns_to_select = [col for col in required_columns if col in available_columns]
    columns_to_select.extend([col for col in optional_columns if col in available_columns])
    
    df = df[columns_to_select]
    df.rename(columns={
        "unique_key": "id",
        "created_date": "created_at",
    }, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"])
    
    print(f"📋 Columns available: {', '.join(df.columns.tolist())}")
    print(f"🏷️  Complaint types found: {df['complaint_type'].nunique()}")
    print(f"📊 Top complaint types:")
    print(df['complaint_type'].value_counts().head(5))

    # Load to Postgres
    engine = create_engine("postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")
    df.to_sql("nyc_311_all_bronx", engine, if_exists="replace", index=False)

    print(f"✅ Successfully saved {len(df)} BRONX complaints to 'nyc_311_all_bronx' table!")
    print(f"📈 Data range: {df['created_at'].min()} to {df['created_at'].max()}")

if __name__ == "__main__":
    main()
