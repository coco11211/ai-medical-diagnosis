"""
Sensor type definitions and simulated sensor implementations
"""
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseSensor(ABC):
    """Base class for all sensors"""

    def __init__(self, sensor_id: str, name: str, location: str):
        self.sensor_id = sensor_id
        self.name = name
        self.location = location
        self.last_reading = None
        self.last_reading_time = None
        self.is_active = True

    @abstractmethod
    def read(self) -> float:
        """Read sensor value"""
        pass

    def get_reading(self) -> Dict[str, Any]:
        """Get formatted sensor reading"""
        value = self.read()
        self.last_reading = value
        self.last_reading_time = datetime.now()

        return {
            'sensor_id': self.sensor_id,
            'name': self.name,
            'location': self.location,
            'value': value,
            'unit': self.get_unit(),
            'timestamp': self.last_reading_time.isoformat(),
            'status': 'active' if self.is_active else 'inactive'
        }

    @abstractmethod
    def get_unit(self) -> str:
        """Get measurement unit"""
        pass


class TemperatureSensor(BaseSensor):
    """Temperature sensor (simulated)"""

    def __init__(self, sensor_id: str, name: str = "Temperature", location: str = "Zone 1"):
        super().__init__(sensor_id, name, location)
        self.baseline = 22.0  # Celsius
        self.variance = 3.0

    def read(self) -> float:
        """Simulate temperature reading (15-30°C typical greenhouse range)"""
        # Add some realistic variation
        value = self.baseline + random.gauss(0, self.variance)
        # Clamp to realistic range
        return max(15.0, min(35.0, value))

    def get_unit(self) -> str:
        return "°C"


class HumiditySensor(BaseSensor):
    """Humidity sensor (simulated)"""

    def __init__(self, sensor_id: str, name: str = "Humidity", location: str = "Zone 1"):
        super().__init__(sensor_id, name, location)
        self.baseline = 65.0  # Percent
        self.variance = 10.0

    def read(self) -> float:
        """Simulate humidity reading (40-80% typical range)"""
        value = self.baseline + random.gauss(0, self.variance)
        # Clamp to 0-100%
        return max(0.0, min(100.0, value))

    def get_unit(self) -> str:
        return "%"


class SoilMoistureSensor(BaseSensor):
    """Soil moisture sensor (simulated)"""

    def __init__(self, sensor_id: str, name: str = "Soil Moisture", location: str = "Bed 1"):
        super().__init__(sensor_id, name, location)
        self.baseline = 50.0  # Percent
        self.variance = 8.0
        self.drainage_rate = 0.1  # Decreases over time

    def read(self) -> float:
        """Simulate soil moisture reading (20-70% typical range)"""
        value = self.baseline + random.gauss(0, self.variance)
        # Clamp to 0-100%
        return max(0.0, min(100.0, value))

    def get_unit(self) -> str:
        return "%"


class LightSensor(BaseSensor):
    """Light intensity sensor (simulated)"""

    def __init__(self, sensor_id: str, name: str = "Light", location: str = "Zone 1"):
        super().__init__(sensor_id, name, location)
        self.baseline = 15000  # Lux
        self.variance = 5000

    def read(self) -> float:
        """Simulate light reading (0-50000 lux typical range)"""
        # Consider time of day for more realistic simulation
        hour = datetime.now().hour

        # Daytime variation (6 AM to 6 PM)
        if 6 <= hour < 18:
            time_factor = 1.0 - abs(12 - hour) / 12  # Peak at noon
            self.baseline = 30000 * time_factor
        else:
            self.baseline = 500  # Night/artificial light

        value = self.baseline + random.gauss(0, self.variance)
        return max(0.0, min(50000.0, value))

    def get_unit(self) -> str:
        return "lux"


class CO2Sensor(BaseSensor):
    """CO2 concentration sensor (simulated)"""

    def __init__(self, sensor_id: str, name: str = "CO2", location: str = "Zone 1"):
        super().__init__(sensor_id, name, location)
        self.baseline = 800  # ppm (typical greenhouse enriched)
        self.variance = 150

    def read(self) -> float:
        """Simulate CO2 reading (400-1500 ppm typical range)"""
        value = self.baseline + random.gauss(0, self.variance)
        return max(400.0, min(2000.0, value))

    def get_unit(self) -> str:
        return "ppm"
