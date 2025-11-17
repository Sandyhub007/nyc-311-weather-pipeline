# scripts/join_311_weather.py

def main():
    import pandas as pd
    from sqlalchemy import create_engine

    # Connect to Postgres
    engine = create_engine("postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")

    # Load both tables
    df_311 = pd.read_sql("SELECT * FROM nyc_311", engine)
    df_weather = pd.read_sql("SELECT * FROM nyc_weather", engine)

    # Prepare time columns
    df_311["created_at_hour"] = pd.to_datetime(df_311["created_at"]).dt.floor("H")
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])

    # Join on rounded time
    df_joined = pd.merge(
        df_311,
        df_weather,
        left_on="created_at_hour",
        right_on="timestamp",
        how="left"
    )

    # Drop extras
    df_joined.drop(columns=["created_at_hour", "timestamp"], inplace=True)

    # Save to Postgres
    df_joined.to_sql("nyc_311_with_weather", engine, if_exists="replace", index=False)

    print("✅ Joined data saved to 'nyc_311_with_weather' in Postgres.")

if __name__ == "__main__":
    main()

