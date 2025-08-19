# scripts/fetch_311_to_postgres.py

def main():
    import requests
    import pandas as pd
    from sqlalchemy import create_engine

    # NYC 311 API endpoint
    API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    params = {"$limit": 500, "$order": "created_date DESC"}

    print("Fetching data from NYC 311 API...")
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data)

    print("Fetched rows:", len(df))

    # Select and rename important columns
    df = df[[
        "unique_key", "created_date", "complaint_type", "descriptor",
        "borough", "latitude", "longitude"
    ]]
    df.rename(columns={
        "unique_key": "id",
        "created_date": "created_at",
    }, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Load to Postgres
    engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/postgres")
    df.to_sql("nyc_311", engine, if_exists="replace", index=False)

    print("✅ 311 data saved to Postgres.")

if __name__ == "__main__":
    main()

