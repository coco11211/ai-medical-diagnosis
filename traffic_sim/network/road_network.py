"""
Road Network - Graph-based road network for traffic simulation
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class RoadType(Enum):
    """Types of roads with different characteristics"""
    HIGHWAY = "highway"
    ARTERIAL = "arterial"
    COLLECTOR = "collector"
    LOCAL = "local"


@dataclass
class Road:
    """Represents a road segment between two intersections"""
    id: str
    start_node: int
    end_node: int
    length: float  # in meters
    lanes: int
    speed_limit: float  # km/h
    road_type: RoadType
    capacity: int  # vehicles per hour
    current_vehicles: int = 0

    @property
    def congestion_level(self) -> float:
        """Calculate current congestion level (0-1)"""
        if self.capacity == 0:
            return 0.0
        return min(1.0, self.current_vehicles / self.capacity)

    @property
    def current_speed(self) -> float:
        """Calculate current average speed based on congestion"""
        # Speed reduces with congestion using BPR function
        congestion = self.congestion_level
        alpha = 0.15
        beta = 4
        speed_factor = 1 / (1 + alpha * (congestion ** beta))
        return self.speed_limit * speed_factor

    @property
    def travel_time(self) -> float:
        """Calculate travel time in seconds"""
        speed_ms = self.current_speed / 3.6  # Convert km/h to m/s
        if speed_ms == 0:
            return float('inf')
        return self.length / speed_ms


@dataclass
class Intersection:
    """Represents a road intersection/node"""
    id: int
    x: float
    y: float
    has_traffic_light: bool = False
    light_cycle: float = 60.0  # seconds


class RoadNetwork:
    """Graph-based road network for traffic simulation"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.roads: Dict[str, Road] = {}
        self.intersections: Dict[int, Intersection] = {}

    def add_intersection(self, intersection: Intersection):
        """Add an intersection to the network"""
        self.intersections[intersection.id] = intersection
        self.graph.add_node(intersection.id,
                          x=intersection.x,
                          y=intersection.y,
                          has_light=intersection.has_traffic_light)

    def add_road(self, road: Road):
        """Add a road segment to the network"""
        self.roads[road.id] = road
        self.graph.add_edge(road.start_node,
                          road.end_node,
                          road_id=road.id,
                          length=road.length,
                          weight=road.travel_time,
                          lanes=road.lanes,
                          speed_limit=road.speed_limit)

    def update_road_weights(self):
        """Update edge weights based on current traffic conditions"""
        for road in self.roads.values():
            if self.graph.has_edge(road.start_node, road.end_node):
                self.graph[road.start_node][road.end_node]['weight'] = road.travel_time

    def get_road(self, start: int, end: int) -> Optional[Road]:
        """Get road between two intersections"""
        if self.graph.has_edge(start, end):
            road_id = self.graph[start][end]['road_id']
            return self.roads.get(road_id)
        return None

    def add_vehicle_to_road(self, start: int, end: int):
        """Add a vehicle to a road segment"""
        road = self.get_road(start, end)
        if road:
            road.current_vehicles += 1
            self.update_road_weights()

    def remove_vehicle_from_road(self, start: int, end: int):
        """Remove a vehicle from a road segment"""
        road = self.get_road(start, end)
        if road and road.current_vehicles > 0:
            road.current_vehicles -= 1
            self.update_road_weights()

    def get_congestion_map(self) -> Dict[str, float]:
        """Get congestion levels for all roads"""
        return {road_id: road.congestion_level
                for road_id, road in self.roads.items()}

    def get_distance(self, node1: int, node2: int) -> float:
        """Calculate Euclidean distance between two nodes"""
        if node1 in self.intersections and node2 in self.intersections:
            i1 = self.intersections[node1]
            i2 = self.intersections[node2]
            return np.sqrt((i1.x - i2.x)**2 + (i1.y - i2.y)**2)
        return 0.0

    def create_grid_network(self, rows: int = 10, cols: int = 10,
                           spacing: float = 200.0) -> 'RoadNetwork':
        """Create a grid-based road network"""
        # Create intersections
        node_id = 0
        node_map = {}

        for i in range(rows):
            for j in range(cols):
                intersection = Intersection(
                    id=node_id,
                    x=j * spacing,
                    y=i * spacing,
                    has_traffic_light=(i % 3 == 0 and j % 3 == 0)
                )
                self.add_intersection(intersection)
                node_map[(i, j)] = node_id
                node_id += 1

        # Create roads (bidirectional)
        road_id = 0
        for i in range(rows):
            for j in range(cols):
                current_node = node_map[(i, j)]

                # Horizontal roads
                if j < cols - 1:
                    next_node = node_map[(i, j + 1)]

                    # Road type varies by position
                    if i % 5 == 0:
                        road_type = RoadType.HIGHWAY
                        lanes = 3
                        speed_limit = 80
                        capacity = 2000
                    elif i % 3 == 0:
                        road_type = RoadType.ARTERIAL
                        lanes = 2
                        speed_limit = 60
                        capacity = 1500
                    else:
                        road_type = RoadType.LOCAL
                        lanes = 1
                        speed_limit = 40
                        capacity = 800

                    # Both directions
                    road1 = Road(
                        id=f"R{road_id}",
                        start_node=current_node,
                        end_node=next_node,
                        length=spacing,
                        lanes=lanes,
                        speed_limit=speed_limit,
                        road_type=road_type,
                        capacity=capacity
                    )
                    self.add_road(road1)
                    road_id += 1

                    road2 = Road(
                        id=f"R{road_id}",
                        start_node=next_node,
                        end_node=current_node,
                        length=spacing,
                        lanes=lanes,
                        speed_limit=speed_limit,
                        road_type=road_type,
                        capacity=capacity
                    )
                    self.add_road(road2)
                    road_id += 1

                # Vertical roads
                if i < rows - 1:
                    next_node = node_map[(i + 1, j)]

                    if j % 5 == 0:
                        road_type = RoadType.HIGHWAY
                        lanes = 3
                        speed_limit = 80
                        capacity = 2000
                    elif j % 3 == 0:
                        road_type = RoadType.ARTERIAL
                        lanes = 2
                        speed_limit = 60
                        capacity = 1500
                    else:
                        road_type = RoadType.LOCAL
                        lanes = 1
                        speed_limit = 40
                        capacity = 800

                    road1 = Road(
                        id=f"R{road_id}",
                        start_node=current_node,
                        end_node=next_node,
                        length=spacing,
                        lanes=lanes,
                        speed_limit=speed_limit,
                        road_type=road_type,
                        capacity=capacity
                    )
                    self.add_road(road1)
                    road_id += 1

                    road2 = Road(
                        id=f"R{road_id}",
                        start_node=next_node,
                        end_node=current_node,
                        length=spacing,
                        lanes=lanes,
                        speed_limit=speed_limit,
                        road_type=road_type,
                        capacity=capacity
                    )
                    self.add_road(road2)
                    road_id += 1

        return self

    def get_network_stats(self) -> Dict:
        """Get statistics about the road network"""
        return {
            'num_intersections': len(self.intersections),
            'num_roads': len(self.roads),
            'total_length_km': sum(r.length for r in self.roads.values()) / 1000,
            'avg_speed_limit': np.mean([r.speed_limit for r in self.roads.values()]),
            'total_capacity': sum(r.capacity for r in self.roads.values()),
            'current_vehicles': sum(r.current_vehicles for r in self.roads.values()),
            'avg_congestion': np.mean([r.congestion_level for r in self.roads.values()])
        }
