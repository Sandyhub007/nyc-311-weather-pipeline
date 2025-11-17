# scripts/fetch_311_full_year.py
# Enhanced script to get a FULL YEAR of data by fetching monthly chunks

def main():
    import requests
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import time

    # NYC 311 API endpoint
    API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    
    # Calculate true 1-year date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)
    
    print(f"🔍 Fetching last 6 months of BRONX data...")
    print(f"📅 Target date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_data = []
    
    # Fetch data in monthly chunks to avoid 50k limit
    current_date = start_date
    month_count = 0
    
    while current_date < end_date:
        month_end = min(current_date + timedelta(days=31), end_date)
        month_count += 1
        
        params = {
            "$limit": 50000,
            "$order": "created_date DESC",
            "$where": (
                f"created_date >= '{current_date.strftime('%Y-%m-%d')}' "
                f"AND created_date < '{month_end.strftime('%Y-%m-%d')}' "
                "AND borough = 'BRONX'"
            ),
            "$select": "unique_key,created_date,complaint_type,descriptor,borough,latitude,longitude,incident_address,city"
        }
        
        print(f"📈 Fetching chunk {month_count}: {current_date.strftime('%Y-%m')} ({current_date.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')})")
        
        try:
            response = requests.get(API_URL, params=params)
            
            if response.status_code != 200:
                print(f"⚠️  Error fetching data for {current_date.strftime('%Y-%m')}: {response.status_code}")
                current_date = month_end
                continue
                
            data = response.json()
            if data:
                all_data.extend(data)
                print(f"   ✅ Got {len(data)} records for {current_date.strftime('%Y-%m')}")
            else:
                print(f"   📭 No data for {current_date.strftime('%Y-%m')}")
                
            # Be nice to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️  Error processing {current_date.strftime('%Y-%m')}: {str(e)}")
        
        current_date = month_end
    
    if not all_data:
        print("❌ No data found for the specified timeframe")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove duplicates (in case of overlapping date ranges)
    df = df.drop_duplicates(subset=['unique_key'], keep='first')
    
    print(f"🎉 Total records collected: {len(df)}")
    print(f"📅 Actual date range: {pd.to_datetime(df['created_date']).min()} to {pd.to_datetime(df['created_date']).max()}")
    
    # Process the data
    required_columns = ["unique_key", "created_date", "complaint_type", "descriptor", "borough"]
    optional_columns = ["latitude", "longitude", "incident_address", "city"]
    
    available_columns = df.columns.tolist()
    columns_to_select = [col for col in required_columns if col in available_columns]
    columns_to_select.extend([col for col in optional_columns if col in available_columns])
    
    df = df[columns_to_select]
    df.rename(columns={
        "unique_key": "id",
        "created_date": "created_at",
    }, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"])
    
    # Show statistics
    print(f"\n📊 DATA SUMMARY:")
    print(f"🏷️  Unique complaint types: {df['complaint_type'].nunique()}")
    print(f"🚗 Illegal parking complaints: {len(df[df['complaint_type'] == 'Illegal Parking'])}")
    print(f"📅 Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"📍 Records with location data: {df['latitude'].notna().sum()}")
    
    print(f"\n🔝 TOP COMPLAINT TYPES:")
    print(df['complaint_type'].value_counts().head(10))
    
    # Save to database
    engine = create_engine("postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")
    df.to_sql("nyc_311_bronx_full_year", engine, if_exists="replace", index=False)
    
    print(f"\n✅ Successfully saved {len(df)} records to 'nyc_311_bronx_full_year' table!")
    print(f"🎯 Ready for advanced time-series analysis in Metabase!")

if __name__ == "__main__":
    main()
