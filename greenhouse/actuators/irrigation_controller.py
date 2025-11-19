"""
Irrigation Control System
Manages watering with soil moisture feedback
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum


class IrrigationMode(Enum):
    """Irrigation modes"""
    MANUAL = 0
    SCHEDULED = 1
    MOISTURE_BASED = 2
    SMART = 3  # ML-based


class WateringZone:
    """Individual watering zone/valve"""

    def __init__(self, zone_id: str, name: str, flow_rate: float = 2.0):
        self.zone_id = zone_id
        self.name = name
        self.flow_rate = flow_rate  # Liters per minute
        self.is_active = False
        self.total_water_used = 0.0  # Liters
        self.last_watering = None
        self.watering_duration = 0  # seconds
        self.watering_start_time = None

    def start_watering(self):
        """Start watering this zone"""
        if not self.is_active:
            self.is_active = True
            self.watering_start_time = datetime.now()
            return f"Started watering zone: {self.name}"
        return f"Zone {self.name} already watering"

    def stop_watering(self):
        """Stop watering this zone"""
        if self.is_active:
            self.is_active = False

            if self.watering_start_time:
                duration = (datetime.now() - self.watering_start_time).total_seconds()
                water_used = (duration / 60) * self.flow_rate
                self.total_water_used += water_used
                self.watering_duration = duration
                self.last_watering = datetime.now()
                self.watering_start_time = None

                return f"Stopped watering zone: {self.name} (Duration: {duration:.1f}s, Water used: {water_used:.2f}L)"
        return f"Zone {self.name} not currently watering"

    def get_status(self) -> Dict[str, Any]:
        """Get zone status"""
        current_session_duration = 0
        if self.is_active and self.watering_start_time:
            current_session_duration = (datetime.now() - self.watering_start_time).total_seconds()

        return {
            'zone_id': self.zone_id,
            'name': self.name,
            'is_active': self.is_active,
            'flow_rate': self.flow_rate,
            'total_water_used': round(self.total_water_used, 2),
            'last_watering': self.last_watering.isoformat() if self.last_watering else None,
            'current_session_duration': round(current_session_duration, 1),
            'timestamp': datetime.now().isoformat()
        }


class IrrigationController:
    """Central irrigation control system"""

    def __init__(self):
        self.zones: Dict[str, WateringZone] = {}
        self.mode = IrrigationMode.MANUAL
        self.moisture_threshold_min = 30  # Percent - start watering
        self.moisture_threshold_max = 60  # Percent - stop watering
        self.schedule: List[Dict[str, Any]] = []
        self.soil_moisture_sensors = {}
        self.auto_irrigation_enabled = False

    def register_zone(self, zone: WateringZone):
        """Register a watering zone"""
        self.zones[zone.zone_id] = zone
        print(f"Registered irrigation zone: {zone.name} ({zone.zone_id})")

    def link_moisture_sensor(self, zone_id: str, sensor_id: str):
        """Link a soil moisture sensor to a zone"""
        if zone_id in self.zones:
            self.soil_moisture_sensors[zone_id] = sensor_id
            print(f"Linked sensor {sensor_id} to zone {zone_id}")

    def start_zone(self, zone_id: str) -> str:
        """Manually start watering a zone"""
        if zone_id in self.zones:
            return self.zones[zone_id].start_watering()
        return f"Zone {zone_id} not found"

    def stop_zone(self, zone_id: str) -> str:
        """Stop watering a zone"""
        if zone_id in self.zones:
            return self.zones[zone_id].stop_watering()
        return f"Zone {zone_id} not found"

    def stop_all_zones(self):
        """Emergency stop - turn off all zones"""
        results = []
        for zone in self.zones.values():
            if zone.is_active:
                result = zone.stop_watering()
                results.append(result)
        if not results:
            return ["No zones currently watering"]
        return results

    def water_zone_duration(self, zone_id: str, duration_seconds: int):
        """Water a zone for specific duration"""
        if zone_id not in self.zones:
            return f"Zone {zone_id} not found"

        zone = self.zones[zone_id]
        zone.start_watering()

        # In real system, this would be handled by a timer/scheduler
        return f"Started watering {zone.name} for {duration_seconds} seconds"

    def add_schedule(self, zone_id: str, time_str: str, duration: int):
        """
        Add scheduled watering
        time_str: Time in HH:MM format
        duration: Duration in seconds
        """
        self.schedule.append({
            'zone_id': zone_id,
            'time': time_str,
            'duration': duration,
            'enabled': True
        })
        return f"Added schedule for zone {zone_id} at {time_str}"

    def apply_schedules(self):
        """Apply scheduled irrigation"""
        current_time = datetime.now().strftime("%H:%M")
        results = []

        for schedule_item in self.schedule:
            if schedule_item['enabled'] and schedule_item['time'] == current_time:
                zone_id = schedule_item['zone_id']
                duration = schedule_item['duration']
                result = self.water_zone_duration(zone_id, duration)
                results.append(result)

        return results if results else ["No schedules to apply at this time"]

    def check_moisture_levels(self, sensor_readings: Dict[str, float]):
        """
        Check moisture levels and activate irrigation if needed
        sensor_readings: Dict of sensor_id -> moisture_value (%)
        """
        if not self.auto_irrigation_enabled:
            return ["Auto irrigation not enabled"]

        results = []

        for zone_id, sensor_id in self.soil_moisture_sensors.items():
            if sensor_id in sensor_readings:
                moisture = sensor_readings[sensor_id]
                zone = self.zones[zone_id]

                # Start watering if below minimum threshold
                if moisture < self.moisture_threshold_min and not zone.is_active:
                    zone.start_watering()
                    results.append(f"Started watering {zone.name} (moisture: {moisture:.1f}%)")

                # Stop watering if above maximum threshold
                elif moisture > self.moisture_threshold_max and zone.is_active:
                    zone.stop_watering()
                    results.append(f"Stopped watering {zone.name} (moisture: {moisture:.1f}%)")

        return results if results else ["All zones within optimal moisture range"]

    def enable_auto_irrigation(self, min_moisture: int = 30, max_moisture: int = 60):
        """Enable moisture-based automatic irrigation"""
        self.auto_irrigation_enabled = True
        self.moisture_threshold_min = min_moisture
        self.moisture_threshold_max = max_moisture
        self.mode = IrrigationMode.MOISTURE_BASED
        return f"Auto irrigation enabled (target: {min_moisture}-{max_moisture}%)"

    def disable_auto_irrigation(self):
        """Disable automatic irrigation"""
        self.auto_irrigation_enabled = False
        self.mode = IrrigationMode.MANUAL
        self.stop_all_zones()
        return "Auto irrigation disabled"

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all zones"""
        total_water_used = sum(zone.total_water_used for zone in self.zones.values())
        active_zones = [z.name for z in self.zones.values() if z.is_active]

        return {
            'mode': self.mode.name,
            'auto_enabled': self.auto_irrigation_enabled,
            'moisture_threshold_min': self.moisture_threshold_min,
            'moisture_threshold_max': self.moisture_threshold_max,
            'total_water_used': round(total_water_used, 2),
            'active_zones': active_zones,
            'zones': {zone_id: zone.get_status() for zone_id, zone in self.zones.items()}
        }

    def get_water_usage_report(self) -> Dict[str, Any]:
        """Generate water usage report"""
        report = {
            'total_water_used': 0,
            'zones': []
        }

        for zone in self.zones.values():
            report['total_water_used'] += zone.total_water_used
            report['zones'].append({
                'zone': zone.name,
                'water_used': round(zone.total_water_used, 2),
                'last_watering': zone.last_watering.isoformat() if zone.last_watering else 'Never'
            })

        report['total_water_used'] = round(report['total_water_used'], 2)
        return report

    def smart_irrigation_recommendation(self,
                                       current_moisture: float,
                                       weather_forecast: str = "sunny",
                                       plant_type: str = "general") -> Dict[str, Any]:
        """
        Provide smart irrigation recommendations based on multiple factors
        This would be enhanced with ML in production
        """
        recommendation = {
            'should_water': False,
            'recommended_duration': 0,
            'reason': ''
        }

        # Simple rule-based logic (would be ML model in production)
        if current_moisture < self.moisture_threshold_min:
            recommendation['should_water'] = True

            # Adjust duration based on moisture deficit
            deficit = self.moisture_threshold_min - current_moisture
            base_duration = 30  # seconds
            recommendation['recommended_duration'] = int(base_duration * (1 + deficit / 30))

            # Adjust for weather
            if weather_forecast.lower() == "rainy":
                recommendation['recommended_duration'] = int(recommendation['recommended_duration'] * 0.5)
                recommendation['reason'] = "Low moisture, but rain expected - reduced watering"
            else:
                recommendation['reason'] = f"Moisture below threshold ({current_moisture:.1f}% < {self.moisture_threshold_min}%)"
        else:
            recommendation['reason'] = f"Moisture adequate ({current_moisture:.1f}%)"

        return recommendation
