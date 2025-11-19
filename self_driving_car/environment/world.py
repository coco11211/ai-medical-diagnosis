"""
PyBullet environment setup with roads, lanes, and obstacles
"""
import pybullet as p
import pybullet_data
import numpy as np
import math


class SimulationWorld:
    """Manages the PyBullet simulation world"""

    def __init__(self, gui=True):
        """Initialize the simulation world"""
        self.gui = gui
        self.client_id = None
        self.plane_id = None
        self.obstacles = []
        self.lane_markers = []

    def setup(self):
        """Setup PyBullet simulation"""
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create road surface
        self._create_road()

        # Create lane markings
        self._create_lane_markings()

        # Add obstacles
        self._create_obstacles()

        return self.client_id

    def _create_road(self):
        """Create a road surface"""
        # Road dimensions
        road_length = 200
        road_width = 8

        # Create dark gray road
        road_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[road_length/2, road_width/2, 0.01]
        )
        road_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[road_length/2, road_width/2, 0.01],
            rgbaColor=[0.2, 0.2, 0.2, 1]
        )

        self.road_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=road_collision,
            baseVisualShapeIndex=road_visual,
            basePosition=[0, 0, 0.01]
        )

    def _create_lane_markings(self):
        """Create white lane markings on the road"""
        road_length = 200
        road_width = 8
        lane_width = road_width / 2  # Two lanes

        # Center line (dashed)
        dash_length = 2
        gap_length = 2
        num_dashes = int(road_length / (dash_length + gap_length))

        for i in range(num_dashes):
            x_pos = -road_length/2 + i * (dash_length + gap_length) + dash_length/2

            # Center dashed line
            marker = self._create_lane_marker(
                position=[x_pos, 0, 0.02],
                size=[dash_length/2, 0.1, 0.005],
                color=[1, 1, 1, 1]
            )
            self.lane_markers.append(marker)

        # Edge lines (solid)
        left_edge = self._create_lane_marker(
            position=[0, road_width/2, 0.02],
            size=[road_length/2, 0.1, 0.005],
            color=[1, 1, 1, 1]
        )
        right_edge = self._create_lane_marker(
            position=[0, -road_width/2, 0.02],
            size=[road_length/2, 0.1, 0.005],
            color=[1, 1, 1, 1]
        )

        self.lane_markers.extend([left_edge, right_edge])

    def _create_lane_marker(self, position, size, color):
        """Create a single lane marker"""
        marker_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=size
        )
        marker_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=size,
            rgbaColor=color
        )

        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=marker_collision,
            baseVisualShapeIndex=marker_visual,
            basePosition=position
        )

        return marker_id

    def _create_obstacles(self):
        """Create obstacles on the road"""
        # Static obstacles (cubes representing other vehicles, cones, etc.)
        obstacle_positions = [
            [20, 2, 0.5],
            [40, -2, 0.5],
            [60, 1.5, 0.5],
            [80, -1.5, 0.5],
            [100, 0, 0.5],
        ]

        for pos in obstacle_positions:
            # Create cube obstacle
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.5, 0.5]
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.5, 0.5],
                rgbaColor=[1, 0, 0, 1]  # Red obstacles
            )

            obstacle_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )

            self.obstacles.append(obstacle_id)

    def add_dynamic_obstacle(self, position, velocity=[0, 0, 0]):
        """Add a moving obstacle"""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.5]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.5],
            rgbaColor=[1, 0.5, 0, 1]  # Orange for dynamic obstacles
        )

        obstacle_id = p.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )

        # Set velocity
        p.resetBaseVelocity(obstacle_id, linearVelocity=velocity)

        self.obstacles.append(obstacle_id)
        return obstacle_id

    def get_obstacles(self):
        """Get all obstacle positions"""
        obstacle_data = []
        for obs_id in self.obstacles:
            pos, orn = p.getBasePositionAndOrientation(obs_id)
            vel, _ = p.getBaseVelocity(obs_id)
            obstacle_data.append({
                'id': obs_id,
                'position': pos,
                'orientation': orn,
                'velocity': vel
            })
        return obstacle_data

    def reset(self):
        """Reset the simulation"""
        if self.client_id is not None:
            p.resetSimulation()
            self.setup()

    def close(self):
        """Close the simulation"""
        if self.client_id is not None:
            p.disconnect()
            self.client_id = None
