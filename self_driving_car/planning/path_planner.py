"""
Path planning using A* and potential field methods
"""
import numpy as np
import heapq
import math
from typing import List, Tuple, Optional


class Node:
    """Node for A* pathfinding"""

    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.position == other.position


class AStarPlanner:
    """A* path planning algorithm"""

    def __init__(self, grid_resolution=0.5):
        """
        Initialize A* planner

        Args:
            grid_resolution: Grid cell size in meters
        """
        self.grid_resolution = grid_resolution

    def plan(self, start, goal, obstacle_grid):
        """
        Plan path from start to goal using A*

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacle_grid: 2D numpy array (1 = obstacle, 0 = free)

        Returns:
            List of waypoints from start to goal
        """
        # Convert positions to grid coordinates
        start_node = Node(self._world_to_grid(start, obstacle_grid.shape))
        goal_node = Node(self._world_to_grid(goal, obstacle_grid.shape))

        # Initialize open and closed lists
        open_list = []
        closed_set = set()

        heapq.heappush(open_list, start_node)

        # Main loop
        while open_list:
            # Get node with lowest f score
            current_node = heapq.heappop(open_list)

            # Check if goal reached
            if current_node == goal_node:
                return self._reconstruct_path(current_node, obstacle_grid.shape)

            closed_set.add(current_node.position)

            # Generate neighbors
            neighbors = self._get_neighbors(current_node, obstacle_grid)

            for neighbor in neighbors:
                if neighbor.position in closed_set:
                    continue

                # Calculate costs
                neighbor.g = current_node.g + self._distance(current_node.position, neighbor.position)
                neighbor.h = self._heuristic(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h

                # Check if neighbor is in open list with higher cost
                if self._in_open_list(neighbor, open_list):
                    continue

                heapq.heappush(open_list, neighbor)

        # No path found
        return None

    def _world_to_grid(self, position, grid_shape):
        """Convert world coordinates to grid coordinates"""
        grid_size = grid_shape[0]
        center = grid_size // 2

        grid_x = int(center + position[0] / self.grid_resolution)
        grid_y = int(center + position[1] / self.grid_resolution)

        return (grid_x, grid_y)

    def _grid_to_world(self, grid_pos, grid_shape):
        """Convert grid coordinates to world coordinates"""
        grid_size = grid_shape[0]
        center = grid_size // 2

        world_x = (grid_pos[0] - center) * self.grid_resolution
        world_y = (grid_pos[1] - center) * self.grid_resolution

        return (world_x, world_y)

    def _get_neighbors(self, node, obstacle_grid):
        """Get valid neighboring nodes"""
        neighbors = []
        grid_shape = obstacle_grid.shape

        # 8-connected grid
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            new_pos = (node.position[0] + dx, node.position[1] + dy)

            # Check bounds
            if not (0 <= new_pos[0] < grid_shape[0] and 0 <= new_pos[1] < grid_shape[1]):
                continue

            # Check obstacle
            if obstacle_grid[new_pos[1], new_pos[0]] > 0:
                continue

            neighbor = Node(new_pos, node)
            neighbors.append(neighbor)

        return neighbors

    def _distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _heuristic(self, pos1, pos2):
        """Heuristic function (Euclidean distance)"""
        return self._distance(pos1, pos2)

    def _in_open_list(self, node, open_list):
        """Check if node is in open list with lower cost"""
        for open_node in open_list:
            if node == open_node and node.g >= open_node.g:
                return True
        return False

    def _reconstruct_path(self, node, grid_shape):
        """Reconstruct path from goal to start"""
        path = []
        current = node

        while current is not None:
            world_pos = self._grid_to_world(current.position, grid_shape)
            path.append(world_pos)
            current = current.parent

        return path[::-1]  # Reverse to get start to goal


class PotentialFieldPlanner:
    """Potential field path planning"""

    def __init__(self, attractive_gain=1.0, repulsive_gain=10.0, influence_distance=5.0):
        """
        Initialize potential field planner

        Args:
            attractive_gain: Gain for attractive force
            repulsive_gain: Gain for repulsive force
            influence_distance: Distance at which obstacles influence the field
        """
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance

    def calculate_attractive_force(self, current_pos, goal_pos):
        """
        Calculate attractive force toward goal

        Args:
            current_pos: Current position [x, y]
            goal_pos: Goal position [x, y]

        Returns:
            Attractive force vector
        """
        diff = np.array(goal_pos) - np.array(current_pos)
        distance = np.linalg.norm(diff)

        if distance < 0.1:
            return np.array([0.0, 0.0])

        # Linear attractive force
        force = self.attractive_gain * diff / distance

        return force

    def calculate_repulsive_force(self, current_pos, obstacles):
        """
        Calculate repulsive force from obstacles

        Args:
            current_pos: Current position [x, y]
            obstacles: List of obstacle positions

        Returns:
            Repulsive force vector
        """
        repulsive_force = np.array([0.0, 0.0])
        current_pos_array = np.array(current_pos)

        for obstacle in obstacles:
            diff = current_pos_array - np.array(obstacle[:2])
            distance = np.linalg.norm(diff)

            if distance < 0.1:
                distance = 0.1

            if distance < self.influence_distance:
                # Inverse square law
                magnitude = self.repulsive_gain * (1.0 / distance - 1.0 / self.influence_distance) / (distance ** 2)
                force = magnitude * diff / distance
                repulsive_force += force

        return repulsive_force

    def calculate_total_force(self, current_pos, goal_pos, obstacles):
        """
        Calculate total potential field force

        Args:
            current_pos: Current position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of obstacle positions

        Returns:
            Total force vector
        """
        attractive = self.calculate_attractive_force(current_pos, goal_pos)
        repulsive = self.calculate_repulsive_force(current_pos, obstacles)

        total_force = attractive + repulsive

        return total_force

    def get_next_position(self, current_pos, goal_pos, obstacles, step_size=0.5):
        """
        Get next position following potential field

        Args:
            current_pos: Current position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of obstacle positions
            step_size: Step size for movement

        Returns:
            Next position
        """
        force = self.calculate_total_force(current_pos, goal_pos, obstacles)

        # Normalize and scale
        force_magnitude = np.linalg.norm(force)
        if force_magnitude < 0.01:
            return current_pos

        direction = force / force_magnitude
        next_pos = np.array(current_pos) + step_size * direction

        return next_pos.tolist()


class PathFollower:
    """Path following controller"""

    def __init__(self, lookahead_distance=2.0):
        """
        Initialize path follower

        Args:
            lookahead_distance: Lookahead distance for pure pursuit
        """
        self.lookahead_distance = lookahead_distance

    def pure_pursuit(self, current_pos, current_heading, path):
        """
        Pure pursuit path following

        Args:
            current_pos: Current position [x, y]
            current_heading: Current heading angle in radians
            path: List of waypoints

        Returns:
            Steering angle and target point
        """
        if not path or len(path) < 2:
            return 0.0, None

        # Find lookahead point
        target_point = self._find_lookahead_point(current_pos, path)

        if target_point is None:
            return 0.0, None

        # Calculate steering angle
        dx = target_point[0] - current_pos[0]
        dy = target_point[1] - current_pos[1]

        target_heading = math.atan2(dy, dx)
        heading_error = self._normalize_angle(target_heading - current_heading)

        # Calculate steering angle (simplified)
        steering_angle = np.clip(heading_error, -0.5, 0.5)

        return steering_angle, target_point

    def _find_lookahead_point(self, current_pos, path):
        """Find lookahead point on path"""
        min_dist = float('inf')
        closest_idx = 0

        # Find closest point on path
        for i, point in enumerate(path):
            dist = math.sqrt((point[0] - current_pos[0])**2 + (point[1] - current_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Find lookahead point
        for i in range(closest_idx, len(path)):
            point = path[i]
            dist = math.sqrt((point[0] - current_pos[0])**2 + (point[1] - current_pos[1])**2)

            if dist >= self.lookahead_distance:
                return point

        # Return last point if no lookahead point found
        return path[-1] if path else None

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
