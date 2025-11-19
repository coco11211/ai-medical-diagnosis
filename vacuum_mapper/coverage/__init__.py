"""
Coverage path planning algorithms for complete area coverage
"""

from .boustrophedon import BoustrophedonPlanner
from .spiral import SpiralPlanner

__all__ = ['BoustrophedonPlanner', 'SpiralPlanner']
