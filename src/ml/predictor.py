"""
Machine learning price prediction module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Machine learning price predictor supporting multiple models.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        lookback_period: int = 60,
        prediction_horizon: int = 5
    ):
        """
        Initialize ML predictor.

        Args:
            model_type: Type of model (lstm, random_forest, gradient_boosting)
            lookback_period: Number of past days to use for prediction
            prediction_horizon: Number of days ahead to predict
        """
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon

        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.

        Args:
            df: DataFrame with market data

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()

        # Price position
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Momentum indicators
        df['roc'] = df['close'].pct_change(periods=10)

        # Lagged features
        for i in range(1, 6):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)

        # Drop NaN values
        df = df.dropna()

        return df

    def create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            data: Input data array
            lookback: Lookback period
            horizon: Prediction horizon

        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []

        for i in range(lookback, len(data) - horizon + 1):
            X.append(data[i - lookback:i])
            y.append(data[i + horizon - 1])

        return np.array(X), np.array(y)

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Training Random Forest model...")

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        # Reshape if needed (flatten sequences)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        logger.info("Random Forest model trained successfully")

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train Gradient Boosting model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Training Gradient Boosting model...")

        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        # Reshape if needed
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        logger.info("Gradient Boosting model trained successfully")

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train LSTM model (requires TensorFlow).

        Args:
            X_train: Training features
            y_train: Training targets
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            logger.info("Training LSTM model...")

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Train model
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            self.model = model
            self.is_trained = True

            logger.info("LSTM model trained successfully")

        except ImportError:
            logger.error("TensorFlow not available. Using Random Forest instead.")
            self.train_random_forest(X_train, y_train)

    def train(
        self,
        df: pd.DataFrame,
        train_split: float = 0.8
    ) -> Dict:
        """
        Train the ML model.

        Args:
            df: DataFrame with market data
            train_split: Train/test split ratio

        Returns:
            Dictionary with training results
        """
        # Prepare features
        df_features = self.prepare_features(df)

        # Select feature columns
        feature_cols = [col for col in df_features.columns
                       if col not in ['open', 'high', 'low', 'close', 'volume']]

        # Prepare data
        features = df_features[feature_cols].values
        target = df_features['close'].values

        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = self.create_sequences(
            features_scaled,
            self.lookback_period,
            self.prediction_horizon
        )

        # Split data
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model based on type
        if self.model_type == "lstm":
            self.train_lstm(X_train, y_train)
        elif self.model_type == "gradient_boosting":
            self.train_gradient_boosting(X_train, y_train)
        else:  # random_forest
            self.train_random_forest(X_train, y_train)

        # Evaluate model
        metrics = self.evaluate(X_test, y_test)

        return {
            'model_type': self.model_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': metrics
        }

    def predict(self, df: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Make predictions.

        Args:
            df: DataFrame with market data
            steps: Number of steps to predict

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        df_features = self.prepare_features(df)

        feature_cols = [col for col in df_features.columns
                       if col not in ['open', 'high', 'low', 'close', 'volume']]

        features = df_features[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)

        # Take last lookback_period samples
        X = features_scaled[-self.lookback_period:].reshape(1, self.lookback_period, -1)

        # Make prediction
        if self.model_type == "lstm":
            predictions_scaled = self.model.predict(X, verbose=0)
        else:
            # Flatten for sklearn models
            X_flat = X.reshape(X.shape[0], -1)
            predictions_scaled = self.model.predict(X_flat)

        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        return predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Make predictions
        if self.model_type == "lstm":
            y_pred = self.model.predict(X_test, verbose=0).flatten()
        else:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred = self.model.predict(X_test_flat)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Inverse transform for interpretable metrics
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

    def get_feature_importance(self, top_n: int = 10) -> Optional[Dict]:
        """
        Get feature importance (for tree-based models).

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained or self.model_type == "lstm":
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # Note: Feature names are lost after sequence creation
            # This returns generic feature indices
            indices = np.argsort(importances)[::-1][:top_n]

            return {
                f'feature_{i}': float(importances[i])
                for i in indices
            }

        return None
