"""
Robot Arm Controller Package
A comprehensive robot arm controller with inverse kinematics, trajectory planning,
vision integration, and collision avoidance.
"""

__version__ = "1.0.0"
__author__ = "Robot Arm Controller"

from .kinematics.inverse_kinematics import InverseKinematics
from .trajectory.path_planner import PathPlanner
from .vision.vision_system import VisionSystem
from .collision.collision_detector import CollisionDetector
from .control.robot_controller import RobotController

__all__ = [
    'InverseKinematics',
    'PathPlanner',
    'VisionSystem',
    'CollisionDetector',
    'RobotController',
]
