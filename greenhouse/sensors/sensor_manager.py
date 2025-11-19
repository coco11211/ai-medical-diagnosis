"""
Sensor Manager - Coordinates all sensor readings
"""
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from .sensor_types import BaseSensor


class SensorManager:
    """Manages all greenhouse sensors"""

    def __init__(self, data_dir: str = "greenhouse_data"):
        self.sensors: Dict[str, BaseSensor] = {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.monitoring = False
        self.monitoring_thread = None
        self.monitoring_interval = 5  # seconds
        self.readings_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def register_sensor(self, sensor: BaseSensor):
        """Register a sensor for monitoring"""
        self.sensors[sensor.sensor_id] = sensor
        print(f"Registered sensor: {sensor.name} ({sensor.sensor_id}) at {sensor.location}")

    def unregister_sensor(self, sensor_id: str):
        """Unregister a sensor"""
        if sensor_id in self.sensors:
            sensor = self.sensors.pop(sensor_id)
            print(f"Unregistered sensor: {sensor.name} ({sensor_id})")

    def read_all_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Read all registered sensors"""
        readings = {}
        timestamp = datetime.now()

        for sensor_id, sensor in self.sensors.items():
            try:
                reading = sensor.get_reading()
                readings[sensor_id] = reading
            except Exception as e:
                print(f"Error reading sensor {sensor_id}: {e}")
                readings[sensor_id] = {
                    'sensor_id': sensor_id,
                    'error': str(e),
                    'timestamp': timestamp.isoformat()
                }

        # Store in history
        self.readings_history.append({
            'timestamp': timestamp.isoformat(),
            'readings': readings
        })

        # Limit history size
        if len(self.readings_history) > self.max_history:
            self.readings_history = self.readings_history[-self.max_history:]

        return readings

    def get_sensor_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get latest reading from specific sensor"""
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].get_reading()
        return None

    def start_monitoring(self, interval: int = 5):
        """Start continuous monitoring"""
        if self.monitoring:
            print("Monitoring already active")
            return

        self.monitoring_interval = interval
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"Started sensor monitoring (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            print("Stopped sensor monitoring")

    def _monitoring_loop(self):
        """Internal monitoring loop"""
        while self.monitoring:
            readings = self.read_all_sensors()
            self._save_readings(readings)
            time.sleep(self.monitoring_interval)

    def _save_readings(self, readings: Dict[str, Any]):
        """Save readings to file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = self.data_dir / f"sensor_data_{timestamp}.jsonl"

        try:
            with open(filename, 'a') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'readings': readings
                }, f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving readings: {e}")

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent readings history"""
        return self.readings_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all sensors"""
        stats = {}

        for sensor_id, sensor in self.sensors.items():
            sensor_readings = []
            for entry in self.readings_history:
                if sensor_id in entry.get('readings', {}):
                    reading = entry['readings'][sensor_id]
                    if 'value' in reading:
                        sensor_readings.append(reading['value'])

            if sensor_readings:
                stats[sensor_id] = {
                    'name': sensor.name,
                    'location': sensor.location,
                    'unit': sensor.get_unit(),
                    'current': sensor_readings[-1] if sensor_readings else None,
                    'min': min(sensor_readings),
                    'max': max(sensor_readings),
                    'avg': sum(sensor_readings) / len(sensor_readings),
                    'count': len(sensor_readings)
                }

        return stats

    def check_alerts(self, thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Check for threshold violations"""
        alerts = []
        readings = self.read_all_sensors()

        for sensor_id, reading in readings.items():
            if sensor_id in thresholds and 'value' in reading:
                value = reading['value']
                threshold = thresholds[sensor_id]

                if 'min' in threshold and value < threshold['min']:
                    alerts.append({
                        'sensor_id': sensor_id,
                        'sensor_name': reading['name'],
                        'alert_type': 'below_minimum',
                        'value': value,
                        'threshold': threshold['min'],
                        'unit': reading['unit'],
                        'timestamp': reading['timestamp']
                    })

                if 'max' in threshold and value > threshold['max']:
                    alerts.append({
                        'sensor_id': sensor_id,
                        'sensor_name': reading['name'],
                        'alert_type': 'above_maximum',
                        'value': value,
                        'threshold': threshold['max'],
                        'unit': reading['unit'],
                        'timestamp': reading['timestamp']
                    })

        return alerts
