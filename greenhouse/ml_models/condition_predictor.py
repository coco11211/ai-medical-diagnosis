"""
Environmental Condition Predictor
Predicts future greenhouse conditions using ML
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple
import pickle
from pathlib import Path
from datetime import datetime, timedelta


class ConditionPredictor:
    """Predicts environmental conditions using machine learning"""

    def __init__(self, model_dir: str = "greenhouse_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.models = {
            'temperature': None,
            'humidity': None,
            'soil_moisture': None,
            'light': None
        }

        self.scalers = {
            'temperature': StandardScaler(),
            'humidity': StandardScaler(),
            'soil_moisture': StandardScaler(),
            'light': StandardScaler()
        }

        self.is_trained = False
        self.feature_names = []

    def prepare_training_data(self, sensor_history: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare sensor history data for training
        sensor_history: List of sensor reading dictionaries
        """
        # Convert to DataFrame
        records = []

        for entry in sensor_history:
            timestamp = pd.to_datetime(entry['timestamp'])
            readings = entry.get('readings', {})

            record = {
                'timestamp': timestamp,
                'hour': timestamp.hour,
                'day_of_week': timestamp.dayofweek,
                'month': timestamp.month
            }

            # Extract sensor values
            for sensor_id, reading in readings.items():
                if 'value' in reading:
                    sensor_name = reading.get('name', '').lower().replace(' ', '_')
                    record[sensor_name] = reading['value']

            records.append(record)

        df = pd.DataFrame(records)

        if df.empty:
            raise ValueError("No data available for training")

        # Sort by timestamp
        df = df.sort_values('timestamp')

        return df

    def create_features(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for ML model"""

        # Time-based features
        features = df[['hour', 'day_of_week', 'month']].copy()

        # Lagged features (previous values)
        if target_col in df.columns:
            for lag in [1, 2, 3, 6, 12]:
                features[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        # Rolling statistics
        if target_col in df.columns:
            features[f'{target_col}_rolling_mean_6'] = df[target_col].rolling(window=6).mean()
            features[f'{target_col}_rolling_std_6'] = df[target_col].rolling(window=6).std()

        # Cross-feature interactions (other environmental factors)
        for col in df.columns:
            if col not in ['timestamp', 'hour', 'day_of_week', 'month', target_col] and pd.api.types.is_numeric_dtype(df[col]):
                features[col] = df[col]

        # Drop rows with NaN values (from lagging and rolling)
        features = features.dropna()

        # Get corresponding target values
        if target_col in df.columns:
            target = df.loc[features.index, target_col].values
        else:
            raise ValueError(f"Target column {target_col} not found in data")

        self.feature_names = features.columns.tolist()

        return features.values, target

    def train(self, sensor_history: List[Dict[str, Any]], model_type: str = 'random_forest'):
        """
        Train prediction models for all conditions
        model_type: 'random_forest' or 'gradient_boosting'
        """
        print("Preparing training data...")
        df = self.prepare_training_data(sensor_history)

        results = {}

        # Train models for each condition
        for condition in ['temperature', 'humidity', 'soil_moisture', 'light']:
            if condition not in df.columns:
                print(f"Warning: {condition} not found in data, skipping")
                continue

            print(f"\nTraining {condition} predictor...")

            try:
                # Create features
                X, y = self.create_features(df, condition)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )

                # Scale features
                X_train_scaled = self.scalers[condition].fit_transform(X_train)
                X_test_scaled = self.scalers[condition].transform(X_test)

                # Train model
                if model_type == 'random_forest':
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                else:  # gradient_boosting
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )

                model.fit(X_train_scaled, y_train)

                # Evaluate
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)

                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)

                results[condition] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'rmse': test_rmse,
                    'mae': test_mae
                }

                print(f"  Train R²: {train_r2:.3f}")
                print(f"  Test R²: {test_r2:.3f}")
                print(f"  RMSE: {test_rmse:.3f}")
                print(f"  MAE: {test_mae:.3f}")

                self.models[condition] = model

            except Exception as e:
                print(f"Error training {condition} model: {e}")

        self.is_trained = True
        return results

    def predict(self, current_conditions: Dict[str, float], hours_ahead: int = 1) -> Dict[str, float]:
        """
        Predict future conditions
        current_conditions: Dict of current sensor values
        hours_ahead: How many hours ahead to predict
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet")

        predictions = {}
        future_time = datetime.now() + timedelta(hours=hours_ahead)

        # Create feature vector
        features = {
            'hour': future_time.hour,
            'day_of_week': future_time.weekday(),
            'month': future_time.month
        }

        # Add current conditions as lagged features
        for condition, value in current_conditions.items():
            features[f'{condition}_lag_1'] = value
            features[f'{condition}_rolling_mean_6'] = value
            features[f'{condition}_rolling_std_6'] = 0.1

        # Predict each condition
        for condition, model in self.models.items():
            if model is not None:
                try:
                    # Create feature vector matching training data
                    feature_vector = np.zeros((1, len(self.feature_names)))

                    for i, feat_name in enumerate(self.feature_names):
                        if feat_name in features:
                            feature_vector[0, i] = features[feat_name]

                    # Scale and predict
                    feature_vector_scaled = self.scalers[condition].transform(feature_vector)
                    prediction = model.predict(feature_vector_scaled)[0]
                    predictions[condition] = float(prediction)

                except Exception as e:
                    print(f"Error predicting {condition}: {e}")
                    predictions[condition] = current_conditions.get(condition, 0.0)

        return predictions

    def save_models(self):
        """Save trained models to disk"""
        for condition, model in self.models.items():
            if model is not None:
                model_path = self.model_dir / f"{condition}_model.pkl"
                scaler_path = self.model_dir / f"{condition}_scaler.pkl"

                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[condition], f)

        # Save feature names
        with open(self.model_dir / "feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)

        print(f"Models saved to {self.model_dir}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            for condition in self.models.keys():
                model_path = self.model_dir / f"{condition}_model.pkl"
                scaler_path = self.model_dir / f"{condition}_scaler.pkl"

                if model_path.exists() and scaler_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[condition] = pickle.load(f)

                    with open(scaler_path, 'rb') as f:
                        self.scalers[condition] = pickle.load(f)

            # Load feature names
            feature_path = self.model_dir / "feature_names.pkl"
            if feature_path.exists():
                with open(feature_path, 'rb') as f:
                    self.feature_names = pickle.load(f)

            self.is_trained = True
            print("Models loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def get_feature_importance(self, condition: str) -> Optional[Dict[str, float]]:
        """Get feature importance for a specific condition model"""
        if condition not in self.models or self.models[condition] is None:
            return None

        model = self.models[condition]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))

        return None
