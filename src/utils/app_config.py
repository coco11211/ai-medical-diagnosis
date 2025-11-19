"""
Application Configuration Manager for Windows
Handles settings storage in Windows AppData
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class AppConfig:
    """Application configuration manager"""

    def __init__(self, app_name: str = "Trading Bot Simulator"):
        self.app_name = app_name
        self.config_dir = self._get_config_directory()
        self.config_file = self.config_dir / 'settings.json'
        self.logger = logging.getLogger(__name__)

        self._ensure_config_directory()
        self._load_or_create_config()

    def _get_config_directory(self) -> Path:
        """Get application configuration directory"""
        if sys.platform == 'win32':
            # Windows: Use AppData/Roaming
            appdata = os.getenv('APPDATA')
            config_dir = Path(appdata) / self.app_name
        elif sys.platform == 'darwin':
            # macOS: Use Application Support
            config_dir = Path.home() / 'Library' / 'Application Support' / self.app_name
        else:
            # Linux: Use .config
            config_dir = Path.home() / '.config' / self.app_name

        return config_dir

    def _ensure_config_directory(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_config(self):
        """Load configuration or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                self.logger.info("Configuration loaded")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                self.config = self._get_default_config()
                self.save()
        else:
            self.config = self._get_default_config()
            self.save()
            self.logger.info("Default configuration created")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'trading': {
                'initial_capital': 100000.0,
                'commission': 0.001,
                'slippage': 0.0005,
            },
            'risk_management': {
                'max_position_size': 0.2,
                'max_portfolio_risk': 0.02,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15,
                'max_drawdown': 0.20,
            },
            'strategies': {
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9,
                },
                'rsi': {
                    'period': 14,
                    'oversold': 30,
                    'overbought': 70,
                },
                'bollinger': {
                    'period': 20,
                    'std_dev': 2,
                },
            },
            'notifications': {
                'trade_executions': True,
                'trading_signals': True,
                'risk_warnings': True,
                'daily_summaries': True,
            },
            'ui': {
                'theme': 'light',
                'window_size': [1200, 800],
                'window_position': None,
                'last_tab': 0,
            },
            'app': {
                'version': '1.0.0',
                'first_run': True,
                'auto_update': True,
                'last_update_check': None,
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Example: config.get('trading.initial_capital')
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation

        Example: config.set('trading.initial_capital', 200000)
        """
        keys = key.split('.')
        target = self.config

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the value
        target[keys[-1]] = value

    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Configuration saved")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self._get_default_config()
        self.save()
        self.logger.info("Configuration reset to defaults")

    def export_config(self, export_path: Optional[Path] = None) -> bool:
        """Export configuration to a file"""
        if not export_path:
            export_path = Path.home() / 'trading_bot_config_export.json'

        try:
            with open(export_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration exported to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting config: {e}")
            return False

    def import_config(self, import_path: Path) -> bool:
        """Import configuration from a file"""
        try:
            with open(import_path, 'r') as f:
                imported_config = json.load(f)

            # Validate imported config has required keys
            default_config = self._get_default_config()
            for key in default_config.keys():
                if key not in imported_config:
                    self.logger.warning(f"Imported config missing key: {key}")
                    imported_config[key] = default_config[key]

            self.config = imported_config
            self.save()
            self.logger.info(f"Configuration imported from {import_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error importing config: {e}")
            return False

    def get_data_directory(self) -> Path:
        """Get application data directory"""
        data_dir = self.config_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def get_cache_directory(self) -> Path:
        """Get application cache directory"""
        if sys.platform == 'win32':
            cache_base = os.getenv('LOCALAPPDATA')
            cache_dir = Path(cache_base) / self.app_name / 'cache'
        else:
            cache_dir = self.config_dir / 'cache'

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
