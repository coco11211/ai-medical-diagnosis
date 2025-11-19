"""
Vehicle Agent - Autonomous vehicle agent with routing and behavior
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class VehicleType(Enum):
    """Types of vehicles with different characteristics"""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"


class VehicleState(Enum):
    """Current state of the vehicle"""
    IDLE = "idle"
    MOVING = "moving"
    WAITING = "waiting"
    ARRIVED = "arrived"


@dataclass
class Vehicle:
    """Agent-based vehicle with autonomous behavior"""
    id: int
    vehicle_type: VehicleType
    origin: int
    destination: int
    current_position: int
    route: List[int] = field(default_factory=list)
    state: VehicleState = VehicleState.IDLE
    spawn_time: float = 0.0
    arrival_time: Optional[float] = None
    route_index: int = 0
    speed: float = 0.0  # m/s
    max_speed: float = 30.0  # m/s
    acceleration: float = 2.0  # m/s²
    position_on_road: float = 0.0  # meters from start of current road segment
    total_distance: float = 0.0
    waiting_time: float = 0.0
    reroute_count: int = 0
    prefer_highways: bool = False

    def __post_init__(self):
        """Initialize vehicle parameters based on type"""
        if self.vehicle_type == VehicleType.CAR:
            self.max_speed = 33.3  # 120 km/h
            self.acceleration = 2.5
            self.prefer_highways = True
        elif self.vehicle_type == VehicleType.TRUCK:
            self.max_speed = 22.2  # 80 km/h
            self.acceleration = 1.5
            self.prefer_highways = True
        elif self.vehicle_type == VehicleType.BUS:
            self.max_speed = 25.0  # 90 km/h
            self.acceleration = 1.8
            self.prefer_highways = False
        elif self.vehicle_type == VehicleType.MOTORCYCLE:
            self.max_speed = 36.1  # 130 km/h
            self.acceleration = 3.5
            self.prefer_highways = True
        elif self.vehicle_type == VehicleType.EMERGENCY:
            self.max_speed = 44.4  # 160 km/h
            self.acceleration = 4.0
            self.prefer_highways = True

    def set_route(self, route: List[int]):
        """Set the vehicle's route"""
        self.route = route
        self.route_index = 0
        if len(route) > 0:
            self.current_position = route[0]
            self.state = VehicleState.MOVING

    def get_next_node(self) -> Optional[int]:
        """Get the next node in the route"""
        if self.route_index + 1 < len(self.route):
            return self.route[self.route_index + 1]
        return None

    def move_to_next_node(self):
        """Move to the next node in the route"""
        if self.route_index + 1 < len(self.route):
            self.route_index += 1
            self.current_position = self.route[self.route_index]
            self.position_on_road = 0.0

            if self.route_index == len(self.route) - 1:
                self.state = VehicleState.ARRIVED
                self.arrival_time = time.time()
        else:
            self.state = VehicleState.ARRIVED

    def update_position(self, dt: float, road_length: float, road_speed_limit: float):
        """
        Update vehicle position along current road segment

        Args:
            dt: Time step in seconds
            road_length: Length of current road in meters
            road_speed_limit: Speed limit in km/h
        """
        if self.state != VehicleState.MOVING:
            return

        # Target speed is minimum of max speed and road speed limit
        target_speed = min(self.max_speed, road_speed_limit / 3.6)

        # Accelerate or decelerate towards target speed
        if self.speed < target_speed:
            self.speed = min(self.speed + self.acceleration * dt, target_speed)
        elif self.speed > target_speed:
            self.speed = max(self.speed - self.acceleration * dt, target_speed)

        # Update position
        distance_traveled = self.speed * dt
        self.position_on_road += distance_traveled
        self.total_distance += distance_traveled

        # Check if reached end of road segment
        if self.position_on_road >= road_length:
            self.move_to_next_node()

    def needs_reroute(self, congestion_threshold: float = 0.7) -> bool:
        """
        Determine if vehicle should reroute based on conditions

        Args:
            congestion_threshold: Congestion level that triggers rerouting
        """
        # Emergency vehicles don't reroute
        if self.vehicle_type == VehicleType.EMERGENCY:
            return False

        # Don't reroute too frequently
        if self.reroute_count >= 3:
            return False

        return False  # Will be determined by congestion prediction

    def get_travel_time(self) -> Optional[float]:
        """Get total travel time in seconds"""
        if self.arrival_time and self.spawn_time:
            return self.arrival_time - self.spawn_time
        return None

    def get_stats(self) -> dict:
        """Get vehicle statistics"""
        travel_time = self.get_travel_time()
        return {
            'id': self.id,
            'type': self.vehicle_type.value,
            'state': self.state.value,
            'origin': self.origin,
            'destination': self.destination,
            'current_position': self.current_position,
            'route_length': len(self.route),
            'route_progress': self.route_index / len(self.route) if self.route else 0,
            'total_distance_km': self.total_distance / 1000,
            'current_speed_kmh': self.speed * 3.6,
            'travel_time_min': travel_time / 60 if travel_time else None,
            'waiting_time_min': self.waiting_time / 60,
            'reroute_count': self.reroute_count
        }


class VehicleManager:
    """Manages all vehicles in the simulation"""

    def __init__(self):
        self.vehicles: dict[int, Vehicle] = {}
        self.next_id = 0
        self.completed_vehicles: List[Vehicle] = []

    def spawn_vehicle(self, vehicle_type: VehicleType, origin: int,
                     destination: int) -> Vehicle:
        """Spawn a new vehicle"""
        vehicle = Vehicle(
            id=self.next_id,
            vehicle_type=vehicle_type,
            origin=origin,
            destination=destination,
            current_position=origin,
            spawn_time=time.time()
        )
        self.vehicles[self.next_id] = vehicle
        self.next_id += 1
        return vehicle

    def remove_vehicle(self, vehicle_id: int):
        """Remove a vehicle from simulation"""
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles.pop(vehicle_id)
            if vehicle.state == VehicleState.ARRIVED:
                self.completed_vehicles.append(vehicle)

    def get_active_vehicles(self) -> List[Vehicle]:
        """Get all active vehicles"""
        return [v for v in self.vehicles.values()
                if v.state != VehicleState.ARRIVED]

    def get_vehicles_on_road(self, start_node: int, end_node: int) -> List[Vehicle]:
        """Get vehicles currently on a specific road"""
        vehicles = []
        for vehicle in self.vehicles.values():
            if vehicle.state == VehicleState.MOVING:
                next_node = vehicle.get_next_node()
                if (vehicle.current_position == start_node and
                    next_node == end_node):
                    vehicles.append(vehicle)
        return vehicles

    def get_stats(self) -> dict:
        """Get statistics for all vehicles"""
        active = self.get_active_vehicles()
        completed = self.completed_vehicles

        if not completed:
            avg_travel_time = 0
            avg_distance = 0
        else:
            travel_times = [v.get_travel_time() for v in completed
                          if v.get_travel_time() is not None]
            avg_travel_time = np.mean(travel_times) if travel_times else 0
            avg_distance = np.mean([v.total_distance for v in completed])

        return {
            'total_spawned': self.next_id,
            'active_vehicles': len(active),
            'completed_vehicles': len(completed),
            'avg_travel_time_min': avg_travel_time / 60,
            'avg_distance_km': avg_distance / 1000,
            'vehicles_by_type': {
                vtype.value: len([v for v in active if v.vehicle_type == vtype])
                for vtype in VehicleType
            },
            'vehicles_by_state': {
                state.value: len([v for v in active if v.state == state])
                for state in VehicleState
            }
        }
