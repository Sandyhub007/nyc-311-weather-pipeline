#!/usr/bin/env python3
"""
Time Series Forecasting Pipeline for NYC 311 Complaints
Implements: Facebook Prophet for daily complaint volume prediction
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class ComplaintForecaster:
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.prophet_model = None

    def load_time_series_data(self):
        """Load daily complaint volume data"""
        query = """
        SELECT
            DATE(created_at) as date,
            COUNT(*) as daily_complaints,
            COUNT(CASE WHEN complaint_type = 'Illegal Parking' THEN 1 END) as illegal_parking,
            COUNT(CASE WHEN complaint_type LIKE '%Noise%' THEN 1 END) as noise_complaints,
            COUNT(CASE WHEN complaint_type LIKE '%Heat%' OR complaint_type LIKE '%Hot Water%' THEN 1 END) as heat_complaints
        FROM nyc_311_bronx_full_year
        WHERE created_at >= NOW() - INTERVAL '180 days'
        GROUP BY DATE(created_at)
        ORDER BY date
        """
        
        print("🔍 Loading time series data from database...")
        df = pd.read_sql(text(query), self.engine)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} days of data")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📊 Average daily complaints: {df['daily_complaints'].mean():.2f}")
        
        return df

    def prepare_prophet_data(self, df, column='daily_complaints'):
        """Prepare data for Prophet forecasting"""
        prophet_df = df[['date', column]].copy()
        prophet_df.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' columns
        
        # Remove any rows with missing values
        prophet_df = prophet_df.dropna()
        
        return prophet_df

    def train_prophet_model(self, df, column='daily_complaints'):
        """Train Prophet model for complaint volume forecasting"""
        print(f"\n🤖 Training Prophet model for {column}...")
        
        prophet_data = self.prepare_prophet_data(df, column)
        
        # Initialize Prophet with custom settings
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.95
        )
        
        # Add custom seasonality for monthly patterns
        self.prophet_model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        print("   Training model...")
        self.prophet_model.fit(prophet_data)
        print("   ✅ Model trained successfully")
        
        return prophet_data

    def forecast_complaints(self, periods=30):
        """Generate forecasts for future periods"""
        if self.prophet_model is None:
            raise ValueError("Model not trained. Call train_prophet_model first.")
        
        print(f"\n🔮 Generating {periods}-day forecast...")
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = self.prophet_model.predict(future)
        
        print(f"   ✅ Forecast generated for {periods} days ahead")
        
        return forecast

    def plot_forecast(self, forecast, actual_data, save_path='reports/forecast_plot.png'):
        """Plot forecast with actual data"""
        print(f"\n📊 Creating forecast visualization...")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot actual data
        ax.plot(actual_data['date'], actual_data['daily_complaints'], 
                label='Actual Complaints', color='blue', linewidth=2, marker='o', markersize=3)
        
        # Plot forecast
        forecast_future = forecast[forecast['ds'] > actual_data['date'].max()]
        ax.plot(forecast['ds'], forecast['yhat'], 
                label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        ax.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'], 
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Add vertical line to show where forecast starts
        ax.axvline(x=actual_data['date'].max(), color='green', 
                  linestyle=':', linewidth=2, label='Forecast Start')
        
        ax.set_title('NYC Bronx Daily Complaint Volume Forecast (Prophet Model)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Complaints', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Forecast plot saved to {save_path}")
        plt.close()

    def plot_components(self, save_path='reports/forecast_components.png'):
        """Plot forecast components (trend, weekly, yearly seasonality)"""
        if self.prophet_model is None:
            return
        
        print(f"\n📊 Creating forecast components visualization...")
        
        fig = self.prophet_model.plot_components(
            self.prophet_model.predict(self.prophet_model.make_future_dataframe(periods=30))
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Components plot saved to {save_path}")
        plt.close()

    def evaluate_forecast_accuracy(self, forecast, actual_data):
        """Calculate forecast accuracy metrics"""
        print(f"\n📏 Evaluating forecast accuracy...")
        
        # Merge forecast with actual data for comparison
        actual_with_forecast = pd.merge(
            actual_data, 
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            left_on='date', 
            right_on='ds', 
            how='inner'
        )
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_with_forecast['daily_complaints'] - actual_with_forecast['yhat']))
        rmse = np.sqrt(np.mean((actual_with_forecast['daily_complaints'] - actual_with_forecast['yhat'])**2))
        mape = np.mean(np.abs((actual_with_forecast['daily_complaints'] - actual_with_forecast['yhat']) / 
                             actual_with_forecast['daily_complaints'])) * 100
        
        print(f"   📊 Mean Absolute Error (MAE): {mae:.2f} complaints/day")
        print(f"   📊 Root Mean Square Error (RMSE): {rmse:.2f} complaints/day")
        print(f"   📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}

    def save_model(self, output_dir='models'):
        """Save trained Prophet model"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.prophet_model:
            model_path = f'{output_dir}/prophet_forecaster.pkl'
            joblib.dump(self.prophet_model, model_path)
            print(f"💾 Prophet model saved to {model_path}")
        else:
            print("⚠️  No model to save. Train the model first.")

    def save_forecast_to_db(self, forecast, table_name='ml_forecast_results'):
        """Save forecast results to database"""
        print(f"\n💾 Saving forecast to database table '{table_name}'...")
        
        # Prepare forecast data
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.columns = ['date', 'predicted_complaints', 'lower_bound', 'upper_bound']
        forecast_df['forecast_date'] = datetime.now()
        
        # Save to database
        forecast_df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        print(f"   ✅ Forecast saved ({len(forecast_df)} rows)")

def main():
    print("="*70)
    print("🔮 NYC 311 TIME SERIES FORECASTING PIPELINE (PROPHET)")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'airflow'),
        'user': os.getenv('POSTGRES_USER', 'airflow'),
        'password': os.getenv('POSTGRES_PASSWORD', 'airflow')
    }
    
    try:
        # Initialize forecaster
        forecaster = ComplaintForecaster(db_config)
        
        # Load time series data
        df = forecaster.load_time_series_data()
        
        # Train Prophet model
        prophet_data = forecaster.train_prophet_model(df, column='daily_complaints')
        
        # Generate forecast for next 30 days
        forecast = forecaster.forecast_complaints(periods=30)
        
        # Evaluate accuracy on historical data
        metrics = forecaster.evaluate_forecast_accuracy(forecast, df)
        
        # Create visualizations
        forecaster.plot_forecast(forecast, df, save_path='/app/reports/forecast_plot.png')
        forecaster.plot_components(save_path='/app/reports/forecast_components.png')
        
        # Save model
        forecaster.save_model(output_dir='/app/models')
        
        # Save forecast to database
        forecaster.save_forecast_to_db(forecast)
        
        print(f"\n{'='*70}")
        print("🎉 FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Finished at: {datetime.now()}")
        
        # Print next week's forecast
        print("\n📅 NEXT 7 DAYS FORECAST:")
        next_week = forecast[forecast['ds'] > df['date'].max()].head(7)
        for _, row in next_week.iterrows():
            print(f"   {row['ds'].strftime('%Y-%m-%d')}: {row['yhat']:.0f} complaints "
                  f"(range: {row['yhat_lower']:.0f} - {row['yhat_upper']:.0f})")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

