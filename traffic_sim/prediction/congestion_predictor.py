"""
Congestion Predictor - ML-based traffic congestion prediction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os


class PredictionModel(Enum):
    """Available prediction models"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    PATTERN_BASED = "pattern_based"


class CongestionPredictor:
    """Predicts traffic congestion using machine learning"""

    def __init__(self, model_type: PredictionModel = PredictionModel.RANDOM_FOREST):
        """
        Initialize congestion predictor

        Args:
            model_type: Type of prediction model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Historical data storage
        self.history_length = 100
        self.road_history: Dict[str, deque] = {}
        self.network_history: deque = deque(maxlen=self.history_length)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the prediction model"""
        if self.model_type == PredictionModel.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == PredictionModel.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

    def update_history(self, road_id: str, congestion_level: float,
                      time_of_day: float, day_of_week: int,
                      weather_factor: float = 1.0):
        """
        Update historical congestion data

        Args:
            road_id: Road identifier
            congestion_level: Current congestion level (0-1)
            time_of_day: Time in hours (0-24)
            day_of_week: Day of week (0-6)
            weather_factor: Weather impact factor (0.5-1.5)
        """
        if road_id not in self.road_history:
            self.road_history[road_id] = deque(maxlen=self.history_length)

        data_point = {
            'congestion': congestion_level,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'weather': weather_factor,
            'timestamp': len(self.road_history[road_id])
        }

        self.road_history[road_id].append(data_point)

    def _extract_features(self, road_id: str, time_of_day: float,
                         day_of_week: int, weather_factor: float = 1.0,
                         lookahead_minutes: int = 15) -> np.ndarray:
        """
        Extract features for prediction

        Args:
            road_id: Road identifier
            time_of_day: Time in hours
            day_of_week: Day of week
            weather_factor: Weather impact
            lookahead_minutes: Minutes to predict ahead

        Returns:
            Feature vector
        """
        features = []

        # Time features
        features.append(time_of_day)
        features.append(day_of_week)
        features.append(np.sin(2 * np.pi * time_of_day / 24))  # Cyclic time
        features.append(np.cos(2 * np.pi * time_of_day / 24))
        features.append(1 if day_of_week >= 5 else 0)  # Weekend flag

        # Weather
        features.append(weather_factor)

        # Lookahead time
        features.append(lookahead_minutes)

        # Historical features
        if road_id in self.road_history and len(self.road_history[road_id]) > 0:
            recent_congestion = [d['congestion'] for d in
                               list(self.road_history[road_id])[-10:]]
            features.append(np.mean(recent_congestion))  # Avg recent congestion
            features.append(np.std(recent_congestion))   # Congestion volatility
            features.append(recent_congestion[-1])       # Latest congestion

            # Pattern features
            if len(recent_congestion) >= 5:
                features.append(recent_congestion[-1] - recent_congestion[-5])  # Trend
            else:
                features.append(0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # No history

        # Rush hour indicators
        is_morning_rush = 1 if 7 <= time_of_day <= 9 else 0
        is_evening_rush = 1 if 17 <= time_of_day <= 19 else 0
        features.append(is_morning_rush)
        features.append(is_evening_rush)

        return np.array(features).reshape(1, -1)

    def train(self, training_data: List[Tuple[str, float, int, float, float]]):
        """
        Train the prediction model

        Args:
            training_data: List of (road_id, time_of_day, day_of_week,
                          weather_factor, congestion_level)
        """
        if len(training_data) < 10:
            print("Insufficient training data")
            return

        X_train = []
        y_train = []

        for road_id, time_of_day, day_of_week, weather, congestion in training_data:
            features = self._extract_features(road_id, time_of_day,
                                            day_of_week, weather, 15)
            X_train.append(features.flatten())
            y_train.append(congestion)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        if self.model:
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            print(f"Model trained with {len(training_data)} samples")

    def predict_congestion(self, road_id: str, time_of_day: float,
                          day_of_week: int, weather_factor: float = 1.0,
                          lookahead_minutes: int = 15) -> float:
        """
        Predict congestion level for a road

        Args:
            road_id: Road identifier
            time_of_day: Time in hours (0-24)
            day_of_week: Day of week (0-6)
            weather_factor: Weather impact factor
            lookahead_minutes: Minutes to predict ahead

        Returns:
            Predicted congestion level (0-1)
        """
        if self.model_type == PredictionModel.PATTERN_BASED:
            return self._pattern_based_prediction(road_id, time_of_day, day_of_week)

        if not self.is_trained:
            # Fallback to pattern-based if not trained
            return self._pattern_based_prediction(road_id, time_of_day, day_of_week)

        # Extract features
        features = self._extract_features(road_id, time_of_day, day_of_week,
                                        weather_factor, lookahead_minutes)
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]

        # Clip to valid range
        return np.clip(prediction, 0.0, 1.0)

    def _pattern_based_prediction(self, road_id: str, time_of_day: float,
                                  day_of_week: int) -> float:
        """
        Simple pattern-based prediction using historical averages

        Args:
            road_id: Road identifier
            time_of_day: Time in hours
            day_of_week: Day of week

        Returns:
            Predicted congestion level
        """
        # Rush hour patterns
        base_congestion = 0.3

        # Morning rush (7-9 AM)
        if 7 <= time_of_day <= 9:
            base_congestion = 0.7 if day_of_week < 5 else 0.4

        # Evening rush (5-7 PM)
        elif 17 <= time_of_day <= 19:
            base_congestion = 0.8 if day_of_week < 5 else 0.5

        # Midday
        elif 11 <= time_of_day <= 14:
            base_congestion = 0.5 if day_of_week < 5 else 0.6

        # Night
        elif time_of_day < 6 or time_of_day > 22:
            base_congestion = 0.1

        # Weekend adjustment
        if day_of_week >= 5:
            if 12 <= time_of_day <= 20:
                base_congestion *= 1.2  # More traffic during weekend day

        # Add some randomness
        noise = np.random.uniform(-0.1, 0.1)
        return np.clip(base_congestion + noise, 0.0, 1.0)

    def predict_network_congestion(self, network, time_of_day: float,
                                   day_of_week: int,
                                   weather_factor: float = 1.0) -> Dict[str, float]:
        """
        Predict congestion for entire network

        Args:
            network: RoadNetwork instance
            time_of_day: Time in hours
            day_of_week: Day of week
            weather_factor: Weather impact

        Returns:
            Dictionary mapping road_id to predicted congestion
        """
        predictions = {}

        for road_id, road in network.roads.items():
            prediction = self.predict_congestion(
                road_id, time_of_day, day_of_week, weather_factor
            )
            predictions[road_id] = prediction

        return predictions

    def get_congestion_hotspots(self, network, threshold: float = 0.7,
                               time_of_day: float = 8.0,
                               day_of_week: int = 2) -> List[str]:
        """
        Identify predicted congestion hotspots

        Args:
            network: RoadNetwork instance
            threshold: Congestion threshold for hotspot
            time_of_day: Time to predict for
            day_of_week: Day to predict for

        Returns:
            List of road IDs predicted to be congested
        """
        predictions = self.predict_network_congestion(
            network, time_of_day, day_of_week
        )

        hotspots = [road_id for road_id, congestion in predictions.items()
                   if congestion >= threshold]

        return hotspots

    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model and self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from trained model"""
        if self.is_trained and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
