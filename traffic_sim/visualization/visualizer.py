"""
Traffic Visualizer - Real-time visualization using matplotlib and pygame
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class VisualizationMode(Enum):
    """Visualization display modes"""
    NETWORK = "network"
    CONGESTION = "congestion"
    ROUTES = "routes"
    HEATMAP = "heatmap"
    LIVE = "live"


class TrafficVisualizer:
    """Real-time traffic simulation visualizer"""

    def __init__(self, network, figsize=(14, 10)):
        """
        Initialize visualizer

        Args:
            network: RoadNetwork instance
            figsize: Figure size for display
        """
        self.network = network
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.animation = None
        self.mode = VisualizationMode.LIVE

        # Color schemes
        self.colors = {
            'road': '#404040',
            'highway': '#606060',
            'arterial': '#505050',
            'local': '#404040',
            'intersection': '#2E86AB',
            'traffic_light': '#FFD166',
            'vehicle_car': '#06FFA5',
            'vehicle_truck': '#EF476F',
            'vehicle_bus': '#FFD166',
            'vehicle_emergency': '#FF0000',
            'low_congestion': '#06FFA5',
            'medium_congestion': '#FFD166',
            'high_congestion': '#EF476F'
        }

    def setup_plot(self):
        """Setup matplotlib figure and axes"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2)
        self.ax.set_title('Traffic Simulation', fontsize=16, pad=20)

    def draw_network(self, show_labels: bool = False):
        """
        Draw the road network

        Args:
            show_labels: Whether to show node labels
        """
        if not self.fig:
            self.setup_plot()

        # Draw roads
        for road in self.network.roads.values():
            start_int = self.network.intersections[road.start_node]
            end_int = self.network.intersections[road.end_node]

            # Road color based on type
            if road.road_type.value == 'highway':
                color = self.colors['highway']
                width = 3
            elif road.road_type.value == 'arterial':
                color = self.colors['arterial']
                width = 2
            else:
                color = self.colors['local']
                width = 1

            self.ax.plot([start_int.x, end_int.x],
                        [start_int.y, end_int.y],
                        color=color, linewidth=width, alpha=0.6, zorder=1)

        # Draw intersections
        for intersection in self.network.intersections.values():
            color = self.colors['traffic_light'] if intersection.has_traffic_light else self.colors['intersection']
            size = 100 if intersection.has_traffic_light else 50

            self.ax.scatter(intersection.x, intersection.y,
                          c=color, s=size, zorder=2, edgecolors='white',
                          linewidths=0.5)

            if show_labels:
                self.ax.text(intersection.x, intersection.y + 30,
                           str(intersection.id),
                           fontsize=8, ha='center', color='white')

    def draw_congestion_map(self):
        """Draw congestion heatmap on roads"""
        if not self.fig:
            self.setup_plot()

        # Draw roads with congestion coloring
        for road in self.network.roads.values():
            start_int = self.network.intersections[road.start_node]
            end_int = self.network.intersections[road.end_node]

            congestion = road.congestion_level

            # Color based on congestion
            if congestion < 0.3:
                color = self.colors['low_congestion']
            elif congestion < 0.7:
                color = self.colors['medium_congestion']
            else:
                color = self.colors['high_congestion']

            # Width based on lanes
            width = road.lanes * 1.5

            self.ax.plot([start_int.x, end_int.x],
                        [start_int.y, end_int.y],
                        color=color, linewidth=width, alpha=0.8, zorder=1)

        # Draw intersections
        for intersection in self.network.intersections.values():
            self.ax.scatter(intersection.x, intersection.y,
                          c=self.colors['intersection'], s=50, zorder=2,
                          edgecolors='white', linewidths=0.5)

        # Add colorbar legend
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'congestion',
            [self.colors['low_congestion'],
             self.colors['medium_congestion'],
             self.colors['high_congestion']]
        )
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax, pad=0.02)
        cbar.set_label('Congestion Level', rotation=270, labelpad=20)

    def draw_vehicles(self, vehicles: List):
        """
        Draw vehicles on the network

        Args:
            vehicles: List of Vehicle objects
        """
        for vehicle in vehicles:
            if vehicle.state.value == 'arrived':
                continue

            # Get vehicle position
            current_node = self.network.intersections.get(vehicle.current_position)
            if not current_node:
                continue

            next_node_id = vehicle.get_next_node()
            if not next_node_id:
                continue

            next_node = self.network.intersections.get(next_node_id)
            if not next_node:
                continue

            # Calculate position along road
            road = self.network.get_road(vehicle.current_position, next_node_id)
            if not road:
                continue

            progress = vehicle.position_on_road / road.length if road.length > 0 else 0
            progress = min(1.0, max(0.0, progress))

            # Interpolate position
            x = current_node.x + (next_node.x - current_node.x) * progress
            y = current_node.y + (next_node.y - current_node.y) * progress

            # Vehicle color based on type
            color_map = {
                'car': self.colors['vehicle_car'],
                'truck': self.colors['vehicle_truck'],
                'bus': self.colors['vehicle_bus'],
                'motorcycle': self.colors['vehicle_car'],
                'emergency': self.colors['vehicle_emergency']
            }
            color = color_map.get(vehicle.vehicle_type.value, self.colors['vehicle_car'])

            # Draw vehicle
            self.ax.scatter(x, y, c=color, s=80, marker='o',
                          zorder=3, edgecolors='white', linewidths=0.5,
                          alpha=0.9)

    def draw_route(self, route: List[int], color: str = '#00FF00',
                  label: str = 'Route', linewidth: float = 3):
        """
        Draw a specific route

        Args:
            route: List of intersection IDs
            color: Color for the route
            label: Label for the route
            linewidth: Width of the route line
        """
        if len(route) < 2:
            return

        x_coords = []
        y_coords = []

        for node_id in route:
            if node_id in self.network.intersections:
                node = self.network.intersections[node_id]
                x_coords.append(node.x)
                y_coords.append(node.y)

        self.ax.plot(x_coords, y_coords, color=color, linewidth=linewidth,
                    alpha=0.8, label=label, zorder=2, linestyle='--')

        # Mark start and end
        self.ax.scatter(x_coords[0], y_coords[0], c='green', s=200,
                      marker='o', zorder=4, edgecolors='white', linewidths=2,
                      label='Start')
        self.ax.scatter(x_coords[-1], y_coords[-1], c='red', s=200,
                      marker='*', zorder=4, edgecolors='white', linewidths=2,
                      label='Destination')

    def update_live(self, vehicles: List, frame: int = 0):
        """
        Update live visualization

        Args:
            vehicles: List of active vehicles
            frame: Frame number (for animation)
        """
        self.ax.clear()

        # Draw base network with congestion
        for road in self.network.roads.values():
            start_int = self.network.intersections[road.start_node]
            end_int = self.network.intersections[road.end_node]

            congestion = road.congestion_level

            # Color based on congestion
            if congestion < 0.3:
                color = self.colors['low_congestion']
            elif congestion < 0.7:
                color = self.colors['medium_congestion']
            else:
                color = self.colors['high_congestion']

            width = 1 + road.lanes * 0.5

            self.ax.plot([start_int.x, end_int.x],
                        [start_int.y, end_int.y],
                        color=color, linewidth=width, alpha=0.6, zorder=1)

        # Draw intersections
        for intersection in self.network.intersections.values():
            color = self.colors['traffic_light'] if intersection.has_traffic_light else self.colors['intersection']
            size = 80 if intersection.has_traffic_light else 40

            self.ax.scatter(intersection.x, intersection.y,
                          c=color, s=size, zorder=2, edgecolors='white',
                          linewidths=0.5)

        # Draw vehicles
        self.draw_vehicles(vehicles)

        # Update title with stats
        active_count = len([v for v in vehicles if v.state.value != 'arrived'])
        avg_speed = np.mean([v.speed * 3.6 for v in vehicles if v.speed > 0]) if vehicles else 0
        network_stats = self.network.get_network_stats()

        title = (f"Traffic Simulation (Frame {frame})\n"
                f"Active Vehicles: {active_count} | "
                f"Avg Speed: {avg_speed:.1f} km/h | "
                f"Avg Congestion: {network_stats['avg_congestion']:.2f}")

        self.ax.set_title(title, fontsize=14, pad=15)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2)

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.colors['low_congestion'], label='Low Congestion'),
            mpatches.Patch(color=self.colors['medium_congestion'], label='Medium Congestion'),
            mpatches.Patch(color=self.colors['high_congestion'], label='High Congestion'),
            mpatches.Patch(color=self.colors['vehicle_car'], label='Vehicles'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def show(self):
        """Display the visualization"""
        if self.fig:
            plt.tight_layout()
            plt.show()

    def save(self, filename: str, dpi: int = 150):
        """
        Save visualization to file

        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        if self.fig:
            plt.tight_layout()
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                       facecolor='#1a1a1a')
            print(f"Visualization saved to {filename}")

    def create_animation(self, update_func, frames: int = 100,
                        interval: int = 100, save_path: Optional[str] = None):
        """
        Create animated visualization

        Args:
            update_func: Function to update visualization each frame
            frames: Number of frames
            interval: Delay between frames in milliseconds
            save_path: Optional path to save animation
        """
        if not self.fig:
            self.setup_plot()

        self.animation = FuncAnimation(
            self.fig, update_func, frames=frames,
            interval=interval, blit=False, repeat=True
        )

        if save_path:
            self.animation.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")

        return self.animation

    def plot_statistics(self, stats_history: List[Dict]):
        """
        Plot simulation statistics over time

        Args:
            stats_history: List of statistics dictionaries
        """
        if not stats_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Traffic Simulation Statistics', fontsize=16)

        # Extract time series data
        times = list(range(len(stats_history)))
        active_vehicles = [s.get('active_vehicles', 0) for s in stats_history]
        avg_speed = [s.get('avg_speed', 0) for s in stats_history]
        avg_congestion = [s.get('avg_congestion', 0) for s in stats_history]
        completed_vehicles = [s.get('completed_vehicles', 0) for s in stats_history]

        # Plot 1: Active Vehicles
        axes[0, 0].plot(times, active_vehicles, color='#06FFA5', linewidth=2)
        axes[0, 0].set_title('Active Vehicles Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Number of Vehicles')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Average Speed
        axes[0, 1].plot(times, avg_speed, color='#FFD166', linewidth=2)
        axes[0, 1].set_title('Average Speed Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Speed (km/h)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Network Congestion
        axes[1, 0].plot(times, avg_congestion, color='#EF476F', linewidth=2)
        axes[1, 0].set_title('Average Congestion Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Congestion Level')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Completed Trips
        axes[1, 1].plot(times, completed_vehicles, color='#2E86AB', linewidth=2)
        axes[1, 1].set_title('Completed Trips Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Number of Trips')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def close(self):
        """Close all plots"""
        if self.fig:
            plt.close(self.fig)
        plt.close('all')
