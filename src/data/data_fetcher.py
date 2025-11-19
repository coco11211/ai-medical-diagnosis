"""
Data fetching module for retrieving market data from various sources.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance")

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches market data from various sources (Yahoo Finance, Alpha Vantage, etc.)
    """

    def __init__(self, source: str = 'yfinance', api_key: Optional[str] = None):
        """
        Initialize data fetcher.

        Args:
            source: Data source ('yfinance', 'alpha_vantage')
            api_key: API key for the data source (if required)
        """
        self.source = source
        self.api_key = api_key
        self.cache = {}

        logger.info(f"DataFetcher initialized with source: {source}")

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"

        # Check cache
        if cache_key in self.cache:
            logger.info(f"Using cached data for {symbol}")
            return self.cache[cache_key].copy()

        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        try:
            if self.source == 'yfinance':
                if not YFINANCE_AVAILABLE:
                    raise ImportError("yfinance is not installed. Install with: pip install yfinance")
                data = self._fetch_yfinance(symbol, start_date, end_date, interval)
            elif self.source == 'alpha_vantage':
                data = self._fetch_alpha_vantage(symbol, start_date, end_date, interval)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")

            # Cache the data
            self.cache[cache_key] = data.copy()

            logger.info(f"Fetched {len(data)} rows for {symbol} from {start_date} to {end_date}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            raise ValueError(f"No data found for {symbol}")

        # Standardize column names
        data.columns = [col.lower() for col in data.columns]

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        return data

    def _fetch_alpha_vantage(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage.
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        # Implementation for Alpha Vantage
        # This would require the alpha_vantage library
        raise NotImplementedError("Alpha Vantage integration not yet implemented")

    def fetch_realtime_data(self, symbol: str) -> dict:
        """
        Fetch real-time market data.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with current price information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            raise

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> dict:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}

        for symbol in symbols:
            try:
                data[symbol] = self.fetch_historical_data(
                    symbol, start_date, end_date, interval
                )
                logger.info(f"Fetched data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                data[symbol] = None

        return data

    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Data cache cleared")

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest closing price
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')

            if hist.empty:
                raise ValueError(f"No data found for {symbol}")

            return hist['Close'].iloc[-1]

        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            raise
