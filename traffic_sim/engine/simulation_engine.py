"""
Traffic Simulation Engine - Main simulation orchestrator
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random

from ..network.road_network import RoadNetwork
from ..agents.vehicle import Vehicle, VehicleType, VehicleState, VehicleManager
from ..optimization.router import Router, RoutingAlgorithm
from ..prediction.congestion_predictor import CongestionPredictor, PredictionModel


@dataclass
class SimulationConfig:
    """Configuration for traffic simulation"""
    # Time settings
    time_step: float = 1.0  # seconds
    simulation_duration: float = 3600.0  # seconds (1 hour)

    # Network settings
    grid_size: Tuple[int, int] = (10, 10)
    spacing: float = 200.0  # meters

    # Vehicle spawning
    spawn_rate: float = 5.0  # vehicles per minute
    vehicle_type_distribution: Dict[VehicleType, float] = None

    # Routing
    routing_algorithm: RoutingAlgorithm = RoutingAlgorithm.A_STAR
    reroute_enabled: bool = True
    reroute_interval: float = 300.0  # seconds

    # Prediction
    use_prediction: bool = True
    prediction_model: PredictionModel = PredictionModel.PATTERN_BASED

    # Visualization
    visualize: bool = True
    update_interval: int = 10  # time steps between visualization updates

    def __post_init__(self):
        """Initialize default values"""
        if self.vehicle_type_distribution is None:
            self.vehicle_type_distribution = {
                VehicleType.CAR: 0.7,
                VehicleType.TRUCK: 0.15,
                VehicleType.BUS: 0.1,
                VehicleType.MOTORCYCLE: 0.04,
                VehicleType.EMERGENCY: 0.01
            }


class TrafficSimulation:
    """Main traffic simulation engine"""

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize traffic simulation

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()

        # Initialize components
        self.network = RoadNetwork()
        self.network.create_grid_network(
            rows=self.config.grid_size[0],
            cols=self.config.grid_size[1],
            spacing=self.config.spacing
        )

        self.vehicle_manager = VehicleManager()
        self.router = Router(self.network)
        self.predictor = CongestionPredictor(
            model_type=self.config.prediction_model
        )

        # Simulation state
        self.current_time = 0.0
        self.running = False
        self.paused = False

        # Statistics tracking
        self.stats_history = []
        self.event_log = []

        # Spawn tracking
        self.last_spawn_time = 0.0
        self.total_spawned = 0

        print(f"Traffic Simulation initialized")
        print(f"Network: {len(self.network.intersections)} intersections, "
              f"{len(self.network.roads)} road segments")

    def spawn_vehicle(self) -> Optional[Vehicle]:
        """
        Spawn a new vehicle at a random origin-destination pair

        Returns:
            Spawned vehicle or None
        """
        # Select random origin and destination
        intersections = list(self.network.intersections.keys())
        if len(intersections) < 2:
            return None

        origin = random.choice(intersections)
        destination = random.choice(intersections)

        # Ensure origin != destination and they're far enough apart
        attempts = 0
        while origin == destination or self.network.get_distance(origin, destination) < 500:
            destination = random.choice(intersections)
            attempts += 1
            if attempts > 20:
                return None

        # Select vehicle type based on distribution
        rand_val = random.random()
        cumulative = 0.0
        vehicle_type = VehicleType.CAR

        for vtype, prob in self.config.vehicle_type_distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                vehicle_type = vtype
                break

        # Spawn vehicle
        vehicle = self.vehicle_manager.spawn_vehicle(
            vehicle_type, origin, destination
        )

        # Calculate route
        route = self.router.find_route(
            origin, destination,
            algorithm=self.config.routing_algorithm,
            avoid_congestion=True
        )

        if route:
            vehicle.set_route(route)
            self.total_spawned += 1
            self.log_event(f"Vehicle {vehicle.id} spawned: {origin} → {destination}")
            return vehicle
        else:
            # Remove vehicle if no route found
            self.vehicle_manager.remove_vehicle(vehicle.id)
            return None

    def update_vehicles(self, dt: float):
        """
        Update all vehicle positions and states

        Args:
            dt: Time step in seconds
        """
        vehicles_to_remove = []

        for vehicle in self.vehicle_manager.get_active_vehicles():
            if vehicle.state == VehicleState.ARRIVED:
                vehicles_to_remove.append(vehicle.id)
                self.log_event(f"Vehicle {vehicle.id} arrived at destination")
                continue

            if vehicle.state != VehicleState.MOVING:
                continue

            # Get current road segment
            next_node = vehicle.get_next_node()
            if not next_node:
                vehicle.state = VehicleState.ARRIVED
                vehicles_to_remove.append(vehicle.id)
                continue

            road = self.network.get_road(vehicle.current_position, next_node)
            if not road:
                continue

            # Update vehicle position
            vehicle.update_position(dt, road.length, road.speed_limit)

            # Check if vehicle moved to next segment
            if vehicle.route_index != vehicle.route_index:
                # Vehicle moved - update road vehicle counts
                self.network.remove_vehicle_from_road(
                    vehicle.current_position, next_node
                )
                new_next = vehicle.get_next_node()
                if new_next:
                    self.network.add_vehicle_to_road(
                        vehicle.current_position, new_next
                    )

        # Remove arrived vehicles
        for vehicle_id in vehicles_to_remove:
            self.vehicle_manager.remove_vehicle(vehicle_id)

    def update_road_vehicles(self):
        """Update vehicle counts on all roads"""
        # Reset all counts
        for road in self.network.roads.values():
            road.current_vehicles = 0

        # Count vehicles on each road
        for vehicle in self.vehicle_manager.get_active_vehicles():
            if vehicle.state == VehicleState.MOVING:
                next_node = vehicle.get_next_node()
                if next_node:
                    self.network.add_vehicle_to_road(
                        vehicle.current_position, next_node
                    )

    def step(self):
        """Execute one simulation time step"""
        dt = self.config.time_step

        # Spawn vehicles based on spawn rate
        spawn_interval = 60.0 / self.config.spawn_rate  # seconds per vehicle
        if self.current_time - self.last_spawn_time >= spawn_interval:
            self.spawn_vehicle()
            self.last_spawn_time = self.current_time

        # Update vehicle positions
        self.update_vehicles(dt)

        # Update road vehicle counts
        self.update_road_vehicles()

        # Update congestion predictions
        if self.config.use_prediction:
            hour_of_day = (self.current_time / 3600) % 24
            day_of_week = int((self.current_time / 86400)) % 7

            for road_id, road in self.network.roads.items():
                self.predictor.update_history(
                    road_id, road.congestion_level,
                    hour_of_day, day_of_week
                )

        # Advance time
        self.current_time += dt

        # Collect statistics
        self.collect_statistics()

    def collect_statistics(self):
        """Collect current simulation statistics"""
        vehicle_stats = self.vehicle_manager.get_stats()
        network_stats = self.network.get_network_stats()

        active_vehicles = self.vehicle_manager.get_active_vehicles()
        avg_speed = np.mean([v.speed * 3.6 for v in active_vehicles]) if active_vehicles else 0

        stats = {
            'time': self.current_time,
            'active_vehicles': vehicle_stats['active_vehicles'],
            'completed_vehicles': vehicle_stats['completed_vehicles'],
            'total_spawned': self.total_spawned,
            'avg_speed': avg_speed,
            'avg_congestion': network_stats['avg_congestion'],
            'avg_travel_time': vehicle_stats['avg_travel_time_min'],
            'vehicles_by_type': vehicle_stats['vehicles_by_type']
        }

        self.stats_history.append(stats)

    def run(self, duration: Optional[float] = None,
           callback: Optional[callable] = None) -> Dict:
        """
        Run simulation

        Args:
            duration: Override simulation duration
            callback: Optional callback function called each step

        Returns:
            Final statistics
        """
        if duration:
            self.config.simulation_duration = duration

        self.running = True
        steps = int(self.config.simulation_duration / self.config.time_step)

        print(f"\nStarting traffic simulation...")
        print(f"Duration: {self.config.simulation_duration/60:.1f} minutes")
        print(f"Time step: {self.config.time_step}s")
        print(f"Spawn rate: {self.config.spawn_rate} vehicles/min")
        print(f"Grid size: {self.config.grid_size}")
        print("-" * 60)

        start_real_time = time.time()

        for step in range(steps):
            if not self.running or self.paused:
                break

            self.step()

            # Progress update
            if step % 100 == 0:
                progress = (step / steps) * 100
                active = len(self.vehicle_manager.get_active_vehicles())
                completed = len(self.vehicle_manager.completed_vehicles)

                print(f"Step {step}/{steps} ({progress:.1f}%) | "
                      f"Active: {active} | Completed: {completed} | "
                      f"Time: {self.current_time/60:.1f}min")

            # Callback
            if callback:
                callback(step, self.vehicle_manager.get_active_vehicles())

        elapsed_real_time = time.time() - start_real_time

        print("-" * 60)
        print(f"Simulation completed in {elapsed_real_time:.2f} seconds")

        # Final statistics
        final_stats = self.get_final_statistics()
        self.print_summary(final_stats)

        return final_stats

    def get_final_statistics(self) -> Dict:
        """Get final simulation statistics"""
        vehicle_stats = self.vehicle_manager.get_stats()
        network_stats = self.network.get_network_stats()

        completed = self.vehicle_manager.completed_vehicles
        travel_times = [v.get_travel_time() for v in completed
                       if v.get_travel_time() is not None]

        return {
            'total_spawned': self.total_spawned,
            'total_completed': len(completed),
            'completion_rate': len(completed) / self.total_spawned if self.total_spawned > 0 else 0,
            'avg_travel_time_min': np.mean(travel_times) / 60 if travel_times else 0,
            'median_travel_time_min': np.median(travel_times) / 60 if travel_times else 0,
            'avg_distance_km': np.mean([v.total_distance for v in completed]) / 1000 if completed else 0,
            'avg_speed_kmh': np.mean([v.total_distance / v.get_travel_time() * 3.6
                                     for v in completed if v.get_travel_time()]) if completed else 0,
            'network_congestion': network_stats['avg_congestion'],
            'simulation_duration_min': self.current_time / 60,
            'vehicles_by_type': vehicle_stats['vehicles_by_type']
        }

    def print_summary(self, stats: Dict):
        """Print simulation summary"""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total Vehicles Spawned:    {stats['total_spawned']}")
        print(f"Completed Trips:           {stats['total_completed']}")
        print(f"Completion Rate:           {stats['completion_rate']*100:.1f}%")
        print(f"Avg Travel Time:           {stats['avg_travel_time_min']:.2f} minutes")
        print(f"Median Travel Time:        {stats['median_travel_time_min']:.2f} minutes")
        print(f"Avg Distance:              {stats['avg_distance_km']:.2f} km")
        print(f"Avg Speed:                 {stats['avg_speed_kmh']:.1f} km/h")
        print(f"Network Congestion:        {stats['network_congestion']:.2%}")
        print(f"Simulation Duration:       {stats['simulation_duration_min']:.1f} minutes")
        print("=" * 60)

    def log_event(self, message: str):
        """Log simulation event"""
        self.event_log.append({
            'time': self.current_time,
            'message': message
        })

    def stop(self):
        """Stop simulation"""
        self.running = False

    def pause(self):
        """Pause simulation"""
        self.paused = True

    def resume(self):
        """Resume simulation"""
        self.paused = False

    def reset(self):
        """Reset simulation to initial state"""
        self.current_time = 0.0
        self.running = False
        self.paused = False
        self.stats_history = []
        self.event_log = []
        self.last_spawn_time = 0.0
        self.total_spawned = 0

        # Clear vehicles
        self.vehicle_manager = VehicleManager()

        # Reset network
        self.network = RoadNetwork()
        self.network.create_grid_network(
            rows=self.config.grid_size[0],
            cols=self.config.grid_size[1],
            spacing=self.config.spacing
        )

        self.router = Router(self.network)

        print("Simulation reset")
