#!/usr/bin/env python3
"""
ML Classification Pipeline for NYC 311 Complaints
Implements: Logistic Regression, Random Forest, and XGBoost
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
import holidays
from pandas.api.types import CategoricalDtype

class ComplaintTypeClassifier:
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.models = {}
        self.encoders = {}
        self.scaler = None

    def load_data(self, sample_size=50000):
        """Load and preprocess training data"""
        query = f"""
        SELECT
            c.id,
            c.created_at,
            c.complaint_type,
            c.descriptor,
            c.borough,
            CAST(c.latitude AS DOUBLE PRECISION) as latitude,
            CAST(c.longitude AS DOUBLE PRECISION) as longitude,
            CAST(w.temperature_c AS DOUBLE PRECISION) as temperature_c,
            CAST(w.precip_mm AS DOUBLE PRECISION) as precip_mm
        FROM nyc_311_bronx_full_year c
        LEFT JOIN nyc_weather w
          ON DATE_TRUNC('hour', c.created_at) = w.timestamp
        WHERE c.complaint_type IS NOT NULL
        AND c.latitude IS NOT NULL
        AND c.longitude IS NOT NULL
        AND c.latitude != ''
        AND c.longitude != ''
        AND c.created_at >= NOW() - INTERVAL '180 days'
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """
        
        print(f"🔍 Loading {sample_size} records from database...")
        df = pd.read_sql(query, self.engine)
        print(f"✅ Loaded {len(df)} records")
        
        return self._preprocess_data(df)

    def _preprocess_data(self, df):
        """Preprocess data for ML"""
        print("🔧 Preprocessing data...")
        
        # Convert to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['temperature_c'] = pd.to_numeric(df.get('temperature_c'), errors='coerce')
        df['precip_mm'] = pd.to_numeric(df.get('precip_mm'), errors='coerce')
        
        # Fill missing weather values
        if df['temperature_c'].notna().any():
            df['temperature_c'] = df['temperature_c'].fillna(df['temperature_c'].median())
        else:
            df['temperature_c'] = df['temperature_c'].fillna(0)
        df['precip_mm'] = df['precip_mm'].fillna(0)
        
        # Extract temporal features
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['month'] = df['created_at'].dt.month
        df['day_of_year'] = df['created_at'].dt.dayofyear
        df['week_of_year'] = df['created_at'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['created_at'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        df['is_commute'] = df['hour'].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)
        df['is_night'] = df['hour'].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)
        
        # Holiday feature
        us_holidays = holidays.US()
        df['is_holiday'] = df['created_at'].dt.date.apply(lambda d: 1 if d in us_holidays else 0)
        
        # Season feature
        season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                      3: 'spring', 4: 'spring', 5: 'spring',
                      6: 'summer', 7: 'summer', 8: 'summer',
                      9: 'fall', 10: 'fall', 11: 'fall'}
        df['season'] = df['month'].map(season_map)
        
        # Temperature / precipitation bins
        df['temp_bucket'] = pd.cut(
            df['temperature_c'], bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['below_freezing', 'cold', 'cool', 'warm', 'hot']
        )
        df['precip_bucket'] = pd.cut(
            df['precip_mm'], bins=[-np.inf, 0.0, 1.0, 5.0, np.inf],
            labels=['dry', 'light', 'moderate', 'heavy']
        )
        
        # Spatial buckets (approx neighborhood)
        df['lat_bucket'] = df['latitude'].round(2)
        df['lon_bucket'] = df['longitude'].round(2)
        df['location_bucket'] = (df['lat_bucket'].astype(str) + "_" + df['lon_bucket'].astype(str))
        
        # Descriptor grouping (first segment before dash or full string)
        df['descriptor_group'] = df['descriptor'].fillna('Unknown').str.split(' - ').str[0]
        
        # Drop rows with missing critical values
        df = df.dropna(subset=['latitude', 'longitude', 'complaint_type'])
        
        # Remove classes with too few samples for reliable training
        MIN_SAMPLES_PER_CLASS = 5
        class_counts = df['complaint_type'].value_counts()
        valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
        dropped_rows = len(df) - len(df[df['complaint_type'].isin(valid_classes)])
        if dropped_rows > 0:
            print(f"⚠️  Dropping {dropped_rows} records from rare complaint types (< {MIN_SAMPLES_PER_CLASS} samples).")
        df = df[df['complaint_type'].isin(valid_classes)].copy()
        
        if df['complaint_type'].nunique() < 2:
            raise ValueError("Not enough complaint types with sufficient samples to train the classifier.")

        # Encode categorical variables
        categorical_mappings = {
            'descriptor': 'descriptor_encoded',
            'descriptor_group': 'descriptor_group_encoded',
            'borough': 'borough_encoded',
            'season': 'season_encoded',
            'temp_bucket': 'temp_bucket_encoded',
            'precip_bucket': 'precip_bucket_encoded',
            'location_bucket': 'location_bucket_encoded'
        }

        for col, encoded_col in categorical_mappings.items():
            series = df[col]
            if isinstance(series.dtype, CategoricalDtype):
                series = series.cat.add_categories(['Unknown']).fillna('Unknown')
            else:
                series = series.fillna('Unknown')
            values = series.astype(str).replace('nan', 'Unknown')
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[encoded_col] = self.encoders[col].fit_transform(values)
            else:
                df[encoded_col] = self.encoders[col].transform(values)

        # Encode target variable (complaint_type)
        if 'complaint_type' not in self.encoders:
            self.encoders['complaint_type'] = LabelEncoder()
            df['complaint_type_encoded'] = self.encoders['complaint_type'].fit_transform(
                df['complaint_type']
            )
        
        # Select features for training
        # Cyclical encodings for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        feature_cols = [
            # Time features
            'hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year', 'quarter',
            'is_weekend', 'is_business_hours', 'is_commute', 'is_night', 'is_holiday',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            # Weather
            'temperature_c', 'precip_mm',
            'temp_bucket_encoded', 'precip_bucket_encoded',
            # Location
            'latitude', 'longitude', 'lat_bucket', 'lon_bucket', 'location_bucket_encoded',
            # Categorical encodings
            'descriptor_encoded', 'descriptor_group_encoded', 'borough_encoded', 'season_encoded'
        ]
        
        result_df = df[feature_cols + ['complaint_type_encoded', 'complaint_type']].copy()
        print(f"✅ Preprocessed {len(result_df)} records with {len(feature_cols)} features")
        
        return result_df

    def train_models(self, df, test_size=0.2):
        """Train all ML models"""
        print("\n🤖 Training ML Models...")
        
        X = df.drop(['complaint_type_encoded', 'complaint_type'], axis=1)
        y = df['complaint_type_encoded']

        # Determine whether stratified split is possible
        class_counts = np.bincount(y)
        min_class_count = class_counts[class_counts > 0].min()
        can_stratify = min_class_count >= 2
        if not can_stratify:
            print("⚠️  Some classes have <2 samples. Proceeding without stratified split.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y if can_stratify else None
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Number of classes: {len(np.unique(y))}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Logistic Regression
        print("\n1️⃣  Training Logistic Regression...")
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr',
            n_jobs=-1
        )
        self.models['logistic'].fit(X_train_scaled, y_train)
        print("   ✅ Logistic Regression trained")
        
        # 2. Random Forest
        print("\n2️⃣  Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=10,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        print("   ✅ Random Forest trained")
        
        # 3. XGBoost
        print("\n3️⃣  Training XGBoost...")
        self.models['xgboost'] = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(self.encoders['complaint_type'].classes_),
            random_state=42,
            max_depth=8,
            learning_rate=0.1,
            n_estimators=100,
            n_jobs=-1
        )
        self.models['xgboost'].fit(X_train, y_train)
        print("   ✅ XGBoost trained")
        
        return X_test, X_test_scaled, y_test

    def evaluate_models(self, X_test, X_test_scaled, y_test):
        """Evaluate all models and generate reports"""
        print("\n📊 Evaluating Models...")
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"📈 {name.upper()} RESULTS")
            print(f"{'='*60}")
            
            if name == 'logistic':
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Get top 10 most common complaint types for detailed report
            top_classes = self.encoders['complaint_type'].inverse_transform(
                np.argsort(np.bincount(y_test))[-10:]
            )
            
            # Generate classification report
            report = classification_report(
                y_test, y_pred,
                target_names=self.encoders['complaint_type'].classes_,
                output_dict=True,
                zero_division=0
            )
            results[name] = {
                'accuracy': accuracy,
                'report': report
            }
            
            # Print summary stats
            print(f"📊 Precision: {report['weighted avg']['precision']:.4f}")
            print(f"📊 Recall: {report['weighted avg']['recall']:.4f}")
            print(f"📊 F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return results

    def save_models(self, output_dir='models'):
        """Save trained models and encoders"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving models to {output_dir}/...")
        
        # Save models
        for name, model in self.models.items():
            model_path = f'{output_dir}/{name}_model.pkl'
            joblib.dump(model, model_path)
            print(f"   ✅ Saved {name} model")
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
            print(f"   ✅ Saved scaler")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            encoder_path = f'{output_dir}/{name}_encoder.pkl'
            joblib.dump(encoder, encoder_path)
            print(f"   ✅ Saved {name} encoder")
        
        print(f"\n✅ All models saved successfully!")

def main():
    print("="*70)
    print("🤖 NYC 311 COMPLAINT TYPE CLASSIFICATION PIPELINE")
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
        # Initialize classifier
        classifier = ComplaintTypeClassifier(db_config)
        
        # Load and preprocess data
        df = classifier.load_data(sample_size=50000)
        
        # Train models
        X_test, X_test_scaled, y_test = classifier.train_models(df)
        
        # Evaluate models
        results = classifier.evaluate_models(X_test, X_test_scaled, y_test)
        
        # Save models
        classifier.save_models(output_dir='/app/models')
        
        print(f"\n{'='*70}")
        print("🎉 CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Finished at: {datetime.now()}")
        
        # Print summary
        print("\n📊 FINAL SUMMARY:")
        for model_name, result in results.items():
            print(f"   {model_name.upper()}: {result['accuracy']*100:.2f}% accuracy")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

