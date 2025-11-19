"""
Configuration management for the trading bot.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration loader and manager."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'trading.initial_capital')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    @property
    def trading(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.config.get('trading', {})

    @property
    def risk_management(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.config.get('risk_management', {})

    @property
    def strategies(self) -> Dict[str, Any]:
        """Get strategies configuration."""
        return self.config.get('strategies', {})

    @property
    def data_sources(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.config.get('data_sources', {})

    @property
    def assets(self) -> Dict[str, Any]:
        """Get assets configuration."""
        return self.config.get('assets', {})

    @property
    def backtesting(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.config.get('backtesting', {})

    @property
    def ml(self) -> Dict[str, Any]:
        """Get machine learning configuration."""
        return self.config.get('ml', {})

    @property
    def alerts(self) -> Dict[str, Any]:
        """Get alerts configuration."""
        return self.config.get('alerts', {})

    @property
    def dashboard(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.config.get('dashboard', {})

    @property
    def paper_trading(self) -> Dict[str, Any]:
        """Get paper trading configuration."""
        return self.config.get('paper_trading', {})
