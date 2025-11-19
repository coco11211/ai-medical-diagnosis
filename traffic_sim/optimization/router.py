"""
Route Optimizer - A*, Dijkstra, and dynamic routing algorithms
"""

import networkx as nx
import numpy as np
from typing import List, Optional, Tuple, Dict
from enum import Enum
import heapq


class RoutingAlgorithm(Enum):
    """Available routing algorithms"""
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    DYNAMIC = "dynamic"
    SHORTEST = "shortest"


class Router:
    """Route optimization with multiple algorithms"""

    def __init__(self, network):
        """
        Initialize router with road network

        Args:
            network: RoadNetwork instance
        """
        self.network = network

    def find_route(self, origin: int, destination: int,
                   algorithm: RoutingAlgorithm = RoutingAlgorithm.A_STAR,
                   avoid_congestion: bool = True) -> Optional[List[int]]:
        """
        Find optimal route between two points

        Args:
            origin: Starting intersection ID
            destination: Destination intersection ID
            algorithm: Routing algorithm to use
            avoid_congestion: Whether to consider current traffic

        Returns:
            List of intersection IDs forming the route, or None if no route exists
        """
        if origin not in self.network.intersections:
            return None
        if destination not in self.network.intersections:
            return None

        # Update weights if considering congestion
        if avoid_congestion:
            self.network.update_road_weights()

        try:
            if algorithm == RoutingAlgorithm.DIJKSTRA:
                return self._dijkstra(origin, destination)
            elif algorithm == RoutingAlgorithm.A_STAR:
                return self._a_star(origin, destination)
            elif algorithm == RoutingAlgorithm.DYNAMIC:
                return self._dynamic_routing(origin, destination)
            elif algorithm == RoutingAlgorithm.SHORTEST:
                return self._shortest_path(origin, destination)
            else:
                return self._a_star(origin, destination)
        except nx.NetworkXNoPath:
            return None

    def _dijkstra(self, origin: int, destination: int) -> List[int]:
        """
        Dijkstra's algorithm for shortest path

        Args:
            origin: Starting node
            destination: Destination node

        Returns:
            List of nodes forming the path
        """
        return nx.dijkstra_path(self.network.graph, origin, destination,
                               weight='weight')

    def _a_star(self, origin: int, destination: int) -> List[int]:
        """
        A* algorithm with Euclidean distance heuristic

        Args:
            origin: Starting node
            destination: Destination node

        Returns:
            List of nodes forming the path
        """
        def heuristic(n1, n2):
            """Euclidean distance heuristic"""
            return self.network.get_distance(n1, n2)

        return nx.astar_path(self.network.graph, origin, destination,
                           heuristic=heuristic, weight='weight')

    def _shortest_path(self, origin: int, destination: int) -> List[int]:
        """
        Shortest path by distance (ignoring congestion)

        Args:
            origin: Starting node
            destination: Destination node

        Returns:
            List of nodes forming the path
        """
        return nx.shortest_path(self.network.graph, origin, destination,
                              weight='length')

    def _dynamic_routing(self, origin: int, destination: int) -> List[int]:
        """
        Dynamic routing that considers real-time traffic and predictions

        Args:
            origin: Starting node
            destination: Destination node

        Returns:
            List of nodes forming the path
        """
        # Use A* with modified weights based on congestion
        congestion_map = self.network.get_congestion_map()

        # Create temporary graph with adjusted weights
        temp_graph = self.network.graph.copy()

        for u, v in temp_graph.edges():
            road = self.network.get_road(u, v)
            if road:
                # Increase weight for congested roads
                congestion_penalty = 1 + (road.congestion_level * 2)
                base_weight = temp_graph[u][v]['weight']
                temp_graph[u][v]['weight'] = base_weight * congestion_penalty

        def heuristic(n1, n2):
            return self.network.get_distance(n1, n2)

        try:
            return nx.astar_path(temp_graph, origin, destination,
                               heuristic=heuristic, weight='weight')
        except nx.NetworkXNoPath:
            return None

    def find_alternative_routes(self, origin: int, destination: int,
                               num_routes: int = 3) -> List[Tuple[List[int], float]]:
        """
        Find multiple alternative routes

        Args:
            origin: Starting intersection
            destination: Destination intersection
            num_routes: Number of alternative routes to find

        Returns:
            List of tuples (route, travel_time)
        """
        routes = []

        try:
            # Primary route
            primary = self._a_star(origin, destination)
            if primary:
                travel_time = self._calculate_route_time(primary)
                routes.append((primary, travel_time))

            # Find alternative routes by temporarily removing edges
            temp_graph = self.network.graph.copy()

            for _ in range(num_routes - 1):
                if len(routes) == 0:
                    break

                # Remove edges from previous route
                last_route = routes[-1][0]
                removed_edges = []

                for i in range(len(last_route) - 1):
                    u, v = last_route[i], last_route[i + 1]
                    if temp_graph.has_edge(u, v):
                        edge_data = temp_graph[u][v].copy()
                        temp_graph.remove_edge(u, v)
                        removed_edges.append((u, v, edge_data))

                # Find alternative route
                try:
                    def heuristic(n1, n2):
                        return self.network.get_distance(n1, n2)

                    alt_route = nx.astar_path(temp_graph, origin, destination,
                                            heuristic=heuristic, weight='weight')
                    travel_time = self._calculate_route_time(alt_route)
                    routes.append((alt_route, travel_time))
                except nx.NetworkXNoPath:
                    pass

                # Restore edges for next iteration
                for u, v, data in removed_edges:
                    temp_graph.add_edge(u, v, **data)

        except nx.NetworkXNoPath:
            pass

        return routes

    def _calculate_route_time(self, route: List[int]) -> float:
        """
        Calculate total travel time for a route

        Args:
            route: List of intersection IDs

        Returns:
            Travel time in seconds
        """
        total_time = 0.0
        for i in range(len(route) - 1):
            road = self.network.get_road(route[i], route[i + 1])
            if road:
                total_time += road.travel_time
        return total_time

    def _calculate_route_distance(self, route: List[int]) -> float:
        """
        Calculate total distance for a route

        Args:
            route: List of intersection IDs

        Returns:
            Distance in meters
        """
        total_distance = 0.0
        for i in range(len(route) - 1):
            road = self.network.get_road(route[i], route[i + 1])
            if road:
                total_distance += road.length
        return total_distance

    def get_route_info(self, route: List[int]) -> Dict:
        """
        Get detailed information about a route

        Args:
            route: List of intersection IDs

        Returns:
            Dictionary with route information
        """
        if not route or len(route) < 2:
            return {
                'valid': False,
                'distance_km': 0,
                'time_min': 0,
                'num_segments': 0,
                'avg_speed_kmh': 0
            }

        distance = self._calculate_route_distance(route)
        travel_time = self._calculate_route_time(route)
        avg_speed = (distance / travel_time * 3.6) if travel_time > 0 else 0

        # Get congestion levels
        congestion_levels = []
        for i in range(len(route) - 1):
            road = self.network.get_road(route[i], route[i + 1])
            if road:
                congestion_levels.append(road.congestion_level)

        return {
            'valid': True,
            'distance_km': distance / 1000,
            'time_min': travel_time / 60,
            'num_segments': len(route) - 1,
            'avg_speed_kmh': avg_speed,
            'avg_congestion': np.mean(congestion_levels) if congestion_levels else 0,
            'max_congestion': max(congestion_levels) if congestion_levels else 0,
            'route_nodes': route
        }

    def get_optimal_departure_time(self, origin: int, destination: int,
                                  time_window: Tuple[float, float],
                                  time_step: float = 300) -> Tuple[float, List[int]]:
        """
        Find optimal departure time within a time window

        Args:
            origin: Starting intersection
            destination: Destination intersection
            time_window: (start_time, end_time) in seconds
            time_step: Time step for checking in seconds

        Returns:
            Tuple of (optimal_departure_time, optimal_route)
        """
        # Simplified version - would need traffic prediction for real implementation
        route = self.find_route(origin, destination, RoutingAlgorithm.A_STAR)
        return (time_window[0], route)
