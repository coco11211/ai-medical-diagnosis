"""
Lighting Control System
Manages grow lights with scheduling and automation
"""
import time
from datetime import datetime, time as dt_time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class LightMode(Enum):
    """Lighting modes"""
    OFF = 0
    ON = 1
    AUTO = 2
    SCHEDULE = 3
    INTENSITY_CONTROL = 4


class GrowLight:
    """Individual grow light controller"""

    def __init__(self, light_id: str, name: str, zone: str, max_intensity: int = 100):
        self.light_id = light_id
        self.name = name
        self.zone = zone
        self.max_intensity = max_intensity
        self.current_intensity = 0
        self.is_on = False
        self.mode = LightMode.OFF
        self.schedule: List[Tuple[dt_time, dt_time, int]] = []  # (start, end, intensity)

    def turn_on(self, intensity: int = 100):
        """Turn light on at specified intensity"""
        self.current_intensity = min(intensity, self.max_intensity)
        self.is_on = True
        self.mode = LightMode.ON
        return f"Light {self.name} turned ON at {self.current_intensity}%"

    def turn_off(self):
        """Turn light off"""
        self.current_intensity = 0
        self.is_on = False
        self.mode = LightMode.OFF
        return f"Light {self.name} turned OFF"

    def set_intensity(self, intensity: int):
        """Set light intensity (0-100%)"""
        self.current_intensity = min(max(0, intensity), self.max_intensity)
        self.is_on = self.current_intensity > 0
        return f"Light {self.name} intensity set to {self.current_intensity}%"

    def get_status(self) -> Dict[str, Any]:
        """Get current light status"""
        return {
            'light_id': self.light_id,
            'name': self.name,
            'zone': self.zone,
            'is_on': self.is_on,
            'intensity': self.current_intensity,
            'max_intensity': self.max_intensity,
            'mode': self.mode.name,
            'timestamp': datetime.now().isoformat()
        }


class LightingController:
    """Central lighting control system"""

    def __init__(self):
        self.lights: Dict[str, GrowLight] = {}
        self.auto_mode_enabled = False
        self.target_light_level = 20000  # lux
        self.light_sensor_reading = 0

    def register_light(self, light: GrowLight):
        """Register a grow light"""
        self.lights[light.light_id] = light
        print(f"Registered light: {light.name} ({light.light_id}) in {light.zone}")

    def turn_on_all(self, intensity: int = 100):
        """Turn on all lights"""
        results = []
        for light in self.lights.values():
            result = light.turn_on(intensity)
            results.append(result)
        return results

    def turn_off_all(self):
        """Turn off all lights"""
        results = []
        for light in self.lights.values():
            result = light.turn_off()
            results.append(result)
        return results

    def set_zone_intensity(self, zone: str, intensity: int):
        """Set intensity for all lights in a zone"""
        results = []
        for light in self.lights.values():
            if light.zone == zone:
                result = light.set_intensity(intensity)
                results.append(result)
        return results

    def set_schedule(self, light_id: str, schedule: List[Tuple[str, str, int]]):
        """
        Set lighting schedule for a light
        schedule: List of (start_time, end_time, intensity) tuples
        Example: [("06:00", "12:00", 80), ("12:00", "18:00", 100), ("18:00", "22:00", 60)]
        """
        if light_id not in self.lights:
            return f"Light {light_id} not found"

        light = self.lights[light_id]
        light.schedule = []

        for start_str, end_str, intensity in schedule:
            start = datetime.strptime(start_str, "%H:%M").time()
            end = datetime.strptime(end_str, "%H:%M").time()
            light.schedule.append((start, end, intensity))

        light.mode = LightMode.SCHEDULE
        return f"Schedule set for light {light.name}"

    def apply_schedules(self):
        """Apply schedules to all scheduled lights"""
        current_time = datetime.now().time()
        results = []

        for light in self.lights.values():
            if light.mode == LightMode.SCHEDULE and light.schedule:
                # Check if current time falls within any schedule
                intensity_set = False

                for start, end, intensity in light.schedule:
                    # Handle schedules that cross midnight
                    if start <= end:
                        if start <= current_time <= end:
                            light.set_intensity(intensity)
                            intensity_set = True
                            break
                    else:  # Crosses midnight
                        if current_time >= start or current_time <= end:
                            light.set_intensity(intensity)
                            intensity_set = True
                            break

                if not intensity_set:
                    light.turn_off()

                results.append(f"{light.name}: {light.current_intensity}%")

        return results

    def auto_adjust(self, current_light_level: float):
        """
        Automatically adjust lighting based on ambient light
        current_light_level: Current ambient light in lux
        """
        self.light_sensor_reading = current_light_level
        results = []

        if not self.auto_mode_enabled:
            return ["Auto mode not enabled"]

        # Calculate needed supplemental lighting
        deficit = max(0, self.target_light_level - current_light_level)

        # Convert deficit to percentage (assuming each light can provide ~5000 lux at 100%)
        needed_intensity = min(100, int((deficit / 5000) * 100))

        for light in self.lights.values():
            if light.mode == LightMode.AUTO:
                light.set_intensity(needed_intensity)
                results.append(f"{light.name}: adjusted to {needed_intensity}%")

        return results

    def enable_auto_mode(self, target_lux: int = 20000):
        """Enable automatic lighting control"""
        self.auto_mode_enabled = True
        self.target_light_level = target_lux

        # Set all lights to auto mode
        for light in self.lights.values():
            light.mode = LightMode.AUTO

        return f"Auto mode enabled (target: {target_lux} lux)"

    def disable_auto_mode(self):
        """Disable automatic lighting control"""
        self.auto_mode_enabled = False
        return "Auto mode disabled"

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all lights"""
        return {
            'auto_mode': self.auto_mode_enabled,
            'target_light_level': self.target_light_level,
            'current_sensor_reading': self.light_sensor_reading,
            'lights': {light_id: light.get_status() for light_id, light in self.lights.items()}
        }

    def get_power_consumption(self) -> float:
        """Estimate total power consumption (watts)"""
        # Assuming each light uses 50W at 100% intensity
        total_watts = 0
        for light in self.lights.values():
            if light.is_on:
                total_watts += (light.current_intensity / 100) * 50
        return total_watts

    def set_photoperiod(self, hours_on: int, start_time: str = "06:00"):
        """
        Set a simple photoperiod for all lights
        hours_on: Number of hours lights should be on
        start_time: When to start the light cycle (HH:MM format)
        """
        start = datetime.strptime(start_time, "%H:%M").time()
        start_hour = start.hour
        end_hour = (start_hour + hours_on) % 24

        end_time = f"{end_hour:02d}:00"

        results = []
        for light_id in self.lights:
            schedule = [(start_time, end_time, 100)]
            result = self.set_schedule(light_id, schedule)
            results.append(result)

        return results
