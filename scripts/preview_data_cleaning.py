import requests
import pandas as pd
from datetime import datetime, timedelta

def preview_cleaning():
    API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    
    # Fetch a small sample of raw data for demonstration
    params = {
        "$limit": 5, 
        "$order": "created_date DESC",
        "$where": "complaint_type = 'Illegal Parking' AND borough = 'BRONX'",
        "$select": "unique_key,created_date,complaint_type,descriptor,borough,latitude,longitude,incident_address,city"
    }

    print("--- RAW API RESPONSE (BEFORE CLEANING) ---")
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        raw_data = response.json()
        for item in raw_data:
            print(item)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching raw data: {e}")
        return

    if not raw_data:
        print("No raw data fetched for preview.")
        return

    print("\n--- PROCESSED DATAFRAME (AFTER INITIAL CLEANING) ---")
    df = pd.DataFrame(raw_data)

    # Apply cleaning steps from fetch_311_full_year.py
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
    
    # Convert created_at to datetime, handling potential errors
    df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce') 

    # Drop duplicates (though unlikely in such a small sample)
    df = df.drop_duplicates(subset=['id'], keep='first')

    print(df.to_string()) # Use to_string() to avoid truncation

if __name__ == "__main__":
    preview_cleaning()
