"""
Data processing module for preparing market data for analysis and trading.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes and prepares market data for analysis and trading.
    """

    def __init__(self):
        """Initialize data processor."""
        logger.info("DataProcessor initialized")

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()

        # Ensure lowercase columns
        df.columns = [col.lower() for col in df.columns]

        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], period=14)

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()

        logger.info(f"Added technical indicators to data with {len(df)} rows")
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            data: DataFrame with OHLC data
            period: ATR period

        Returns:
            ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def normalize_data(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize data using min-max scaling.

        Args:
            data: DataFrame to normalize
            columns: Columns to normalize (if None, normalize all numeric columns)

        Returns:
            Normalized DataFrame
        """
        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()

                if max_val - min_val != 0:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_normalized'] = 0

        logger.info(f"Normalized {len(columns)} columns")
        return df

    def resample_data(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample data to a different frequency.

        Args:
            data: DataFrame with time-indexed data
            freq: Target frequency ('1H', '4H', '1D', '1W', etc.)

        Returns:
            Resampled DataFrame
        """
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex for resampling")

        # Resample OHLCV data
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        logger.info(f"Resampled data to {freq} frequency: {len(resampled)} rows")
        return resampled

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing NaN values and outliers.

        Args:
            data: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = data.copy()

        # Remove rows with any NaN in OHLCV columns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        existing_cols = [col for col in ohlcv_cols if col in df.columns]

        initial_rows = len(df)
        df = df.dropna(subset=existing_cols)

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with NaN values")

        # Remove outliers (prices beyond 3 standard deviations)
        if 'close' in df.columns:
            mean = df['close'].mean()
            std = df['close'].std()
            df = df[np.abs(df['close'] - mean) <= (3 * std)]

        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df

    def add_lag_features(self, data: pd.DataFrame, columns: List[str], lags: int = 5) -> pd.DataFrame:
        """
        Add lagged features for machine learning.

        Args:
            data: DataFrame
            columns: Columns to create lags for
            lags: Number of lag periods

        Returns:
            DataFrame with lag features
        """
        df = data.copy()

        for col in columns:
            if col in df.columns:
                for lag in range(1, lags + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        logger.info(f"Added {lags} lag features for {len(columns)} columns")
        return df

    def calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate returns over multiple periods.

        Args:
            data: DataFrame with price data
            periods: List of periods to calculate returns for

        Returns:
            DataFrame with returns
        """
        df = data.copy()

        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)

        logger.info(f"Calculated returns for {len(periods)} periods")
        return df

    def detect_trend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market trend.

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with trend indicators
        """
        df = data.copy()

        # Price above/below moving averages
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['trend_sma'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)

        # MACD trend
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['trend_macd'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

        # Price momentum
        if 'close' in df.columns:
            df['momentum'] = df['close'].pct_change(periods=20)
            df['trend_momentum'] = np.where(df['momentum'] > 0, 1, -1)

        logger.info("Added trend detection indicators")
        return df

    def prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features for machine learning.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with ML features
        """
        df = data.copy()

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Add lag features for important columns
        lag_columns = ['close', 'volume', 'rsi', 'macd']
        df = self.add_lag_features(df, lag_columns, lags=5)

        # Add returns
        df = self.calculate_returns(df)

        # Add trend detection
        df = self.detect_trend(df)

        # Clean data
        df = self.clean_data(df)

        logger.info(f"Prepared ML features: {df.shape[1]} features for {len(df)} samples")
        return df

    def split_train_test(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> tuple:
        """
        Split data into training and testing sets.

        Args:
            data: DataFrame to split
            test_size: Proportion of data for testing
            shuffle: Whether to shuffle data (not recommended for time series)

        Returns:
            Tuple of (train_data, test_data)
        """
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data
