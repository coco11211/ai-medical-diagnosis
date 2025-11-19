"""
Actuator Manager - Coordinates all actuators
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from .lighting_controller import LightingController
from .irrigation_controller import IrrigationController


class ActuatorManager:
    """Central manager for all greenhouse actuators"""

    def __init__(self, data_dir: str = "greenhouse_data"):
        self.lighting = LightingController()
        self.irrigation = IrrigationController()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.automation_enabled = False

    def get_full_status(self) -> Dict[str, Any]:
        """Get status of all actuators"""
        return {
            'timestamp': datetime.now().isoformat(),
            'automation_enabled': self.automation_enabled,
            'lighting': self.lighting.get_all_status(),
            'irrigation': self.irrigation.get_all_status(),
            'power_consumption_watts': self.lighting.get_power_consumption(),
            'water_usage': self.irrigation.get_water_usage_report()
        }

    def enable_full_automation(self):
        """Enable complete automation"""
        self.automation_enabled = True
        self.lighting.enable_auto_mode()
        self.irrigation.enable_auto_irrigation()
        return "Full automation enabled"

    def disable_full_automation(self):
        """Disable automation - manual control only"""
        self.automation_enabled = False
        self.lighting.disable_auto_mode()
        self.irrigation.disable_auto_irrigation()
        return "Full automation disabled"

    def process_sensor_data(self, sensor_readings: Dict[str, Dict[str, Any]]):
        """
        Process sensor data and adjust actuators accordingly
        sensor_readings: Dictionary of sensor data from SensorManager
        """
        results = {
            'lighting_actions': [],
            'irrigation_actions': []
        }

        # Extract light level for lighting control
        light_level = None
        for sensor_id, reading in sensor_readings.items():
            if reading.get('name') == 'Light' and 'value' in reading:
                light_level = reading['value']
                break

        if light_level is not None and self.lighting.auto_mode_enabled:
            lighting_results = self.lighting.auto_adjust(light_level)
            results['lighting_actions'] = lighting_results

        # Extract soil moisture for irrigation control
        moisture_readings = {}
        for sensor_id, reading in sensor_readings.items():
            if 'Soil Moisture' in reading.get('name', '') and 'value' in reading:
                moisture_readings[sensor_id] = reading['value']

        if moisture_readings and self.irrigation.auto_irrigation_enabled:
            irrigation_results = self.irrigation.check_moisture_levels(moisture_readings)
            results['irrigation_actions'] = irrigation_results

        # Apply schedules
        results['lighting_schedule'] = self.lighting.apply_schedules()
        results['irrigation_schedule'] = self.irrigation.apply_schedules()

        # Log actions
        self._log_actions(results)

        return results

    def _log_actions(self, actions: Dict[str, Any]):
        """Log actuator actions"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = self.data_dir / f"actuator_log_{timestamp}.jsonl"

        try:
            with open(filename, 'a') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'actions': actions
                }, f)
                f.write('\n')
        except Exception as e:
            print(f"Error logging actions: {e}")

    def emergency_stop(self):
        """Emergency stop all actuators"""
        self.lighting.turn_off_all()
        self.irrigation.stop_all_zones()
        return "Emergency stop executed - all actuators stopped"

    def night_mode(self):
        """Set to night mode"""
        self.lighting.turn_off_all()
        return "Night mode activated"

    def day_mode(self, light_intensity: int = 80):
        """Set to day mode"""
        self.lighting.turn_on_all(light_intensity)
        return f"Day mode activated (intensity: {light_intensity}%)"
