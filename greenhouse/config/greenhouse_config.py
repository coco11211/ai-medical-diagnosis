"""
Greenhouse Configuration Manager
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


class GreenhouseConfig:
    """Manages greenhouse configuration"""

    def __init__(self, config_file: str = "greenhouse_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_default_config()

        if self.config_file.exists():
            self.load()
        else:
            self.save()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'greenhouse': {
                'name': 'Smart Greenhouse',
                'location': 'Zone 1',
                'timezone': 'UTC'
            },

            'sensors': {
                'monitoring_interval': 5,  # seconds
                'data_retention_days': 30,
                'thresholds': {
                    'temperature': {
                        'min': 15,
                        'max': 30,
                        'optimal': 22
                    },
                    'humidity': {
                        'min': 40,
                        'max': 80,
                        'optimal': 60
                    },
                    'soil_moisture': {
                        'min': 30,
                        'max': 70,
                        'optimal': 50
                    },
                    'light': {
                        'min': 10000,
                        'max': 40000,
                        'optimal': 25000
                    },
                    'co2': {
                        'min': 400,
                        'max': 1500,
                        'optimal': 1000
                    }
                }
            },

            'lighting': {
                'auto_mode': True,
                'target_light_level': 20000,  # lux
                'photoperiod_hours': 16,
                'start_time': '06:00',
                'default_intensity': 100,  # percent
                'energy_saving_mode': False
            },

            'irrigation': {
                'auto_mode': True,
                'moisture_threshold_min': 35,  # percent
                'moisture_threshold_max': 60,  # percent
                'default_duration': 60,  # seconds
                'max_daily_water': 50,  # liters
                'schedule': [
                    {'time': '07:00', 'duration': 60},
                    {'time': '19:00', 'duration': 45}
                ]
            },

            'ml_prediction': {
                'enabled': True,
                'model_type': 'random_forest',  # or 'gradient_boosting'
                'prediction_horizon': 6,  # hours
                'retrain_interval_days': 7,
                'min_training_samples': 100
            },

            'alerts': {
                'enabled': True,
                'daily_summary': True,
                'daily_summary_time': '20:00',
                'critical_alerts_only': False,
                'notification_sound': True
            },

            'dashboard': {
                'host': '127.0.0.1',
                'port': 8050,
                'auto_refresh_seconds': 5,
                'theme': 'light'
            },

            'plant_settings': {
                'plant_type': 'general',
                'planting_date': None,
                'expected_harvest_days': 90,
                'target_yield_kg': 2.0
            },

            'automation': {
                'full_automation': False,
                'override_enabled': True,
                'safety_limits': {
                    'max_temperature': 35,
                    'min_temperature': 10,
                    'max_watering_duration': 300  # seconds
                }
            },

            'data_storage': {
                'data_directory': 'greenhouse_data',
                'model_directory': 'greenhouse_models',
                'backup_enabled': True,
                'backup_interval_days': 1
            }
        }

    def load(self) -> bool:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)

            # Merge with defaults (in case new settings were added)
            self.config = self._merge_configs(self.config, loaded_config)

            print(f"Configuration loaded from {self.config_file}")
            return True

        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def save(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

            print(f"Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Merge loaded config with default config"""
        result = default.copy()

        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path
        Example: config.get('sensors.monitoring_interval')
        """
        keys = path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, path: str, value: Any) -> bool:
        """
        Set configuration value by path
        Example: config.set('sensors.monitoring_interval', 10)
        """
        keys = path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        return True

    def get_sensor_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get sensor thresholds"""
        return self.get('sensors.thresholds', {})

    def get_lighting_config(self) -> Dict[str, Any]:
        """Get lighting configuration"""
        return self.get('lighting', {})

    def get_irrigation_config(self) -> Dict[str, Any]:
        """Get irrigation configuration"""
        return self.get('irrigation', {})

    def is_automation_enabled(self) -> bool:
        """Check if full automation is enabled"""
        return self.get('automation.full_automation', False)

    def export_json(self, filename: str):
        """Export configuration as JSON"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return False

    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("GREENHOUSE CONFIGURATION")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False))
        print("="*60 + "\n")
