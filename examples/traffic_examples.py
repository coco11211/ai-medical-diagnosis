"""
Traffic Simulator Examples

Demonstrates various features of the traffic simulator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from traffic_sim.engine.simulation_engine import TrafficSimulation, SimulationConfig
from traffic_sim.visualization.visualizer import TrafficVisualizer
from traffic_sim.optimization.router import RoutingAlgorithm
from traffic_sim.prediction.congestion_predictor import PredictionModel
from traffic_sim.agents.vehicle import VehicleType


def example1_basic_simulation():
    """Example 1: Basic traffic simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Traffic Simulation")
    print("="*70)

    # Create configuration
    config = SimulationConfig(
        time_step=1.0,
        simulation_duration=600,  # 10 minutes
        grid_size=(8, 8),
        spawn_rate=8.0,  # 8 vehicles per minute
        visualize=True
    )

    # Create and run simulation
    sim = TrafficSimulation(config)

    print("\nRunning simulation...")
    results = sim.run()

    # Visualize final state
    visualizer = TrafficVisualizer(sim.network)
    visualizer.setup_plot()
    visualizer.draw_congestion_map()
    plt.title('Final Network Congestion State')
    plt.savefig('example1_congestion.png', dpi=150)
    print("Saved: example1_congestion.png")

    # Plot statistics
    fig = visualizer.plot_statistics(sim.stats_history)
    plt.savefig('example1_statistics.png', dpi=150)
    print("Saved: example1_statistics.png")

    plt.show()


def example2_route_comparison():
    """Example 2: Compare routing algorithms"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Route Optimization Comparison")
    print("="*70)

    # Create simulation
    config = SimulationConfig(grid_size=(12, 12), spacing=250)
    sim = TrafficSimulation(config)

    # Add some congestion to make algorithms differ
    import random
    for road in random.sample(list(sim.network.roads.values()), 20):
        road.current_vehicles = int(road.capacity * 0.8)

    sim.network.update_road_weights()

    # Select origin and destination
    origin = 0
    destination = 143  # Far corner

    print(f"\nComparing routes from node {origin} to node {destination}")

    # Test different algorithms
    algorithms = {
        'Shortest Path': RoutingAlgorithm.SHORTEST,
        'Dijkstra': RoutingAlgorithm.DIJKSTRA,
        'A* Search': RoutingAlgorithm.A_STAR,
        'Dynamic (Congestion-Aware)': RoutingAlgorithm.DYNAMIC
    }

    # Find routes
    routes = {}
    for name, algo in algorithms.items():
        route = sim.router.find_route(origin, destination, algorithm=algo)
        if route:
            info = sim.router.get_route_info(route)
            routes[name] = (route, info)

            print(f"\n{name}:")
            print(f"  Distance: {info['distance_km']:.2f} km")
            print(f"  Est. Time: {info['time_min']:.2f} min")
            print(f"  Avg Speed: {info['avg_speed_kmh']:.1f} km/h")
            print(f"  Avg Congestion: {info['avg_congestion']:.2%}")
            print(f"  Route Length: {len(route)} nodes")

    # Visualize
    visualizer = TrafficVisualizer(sim.network, figsize=(16, 12))
    visualizer.setup_plot()
    visualizer.draw_congestion_map()

    colors = ['#00FF00', '#00FFFF', '#FF00FF', '#FFFF00']
    for (name, (route, info)), color in zip(routes.items(), colors):
        visualizer.draw_route(route, color, name, linewidth=2.5)

    plt.legend(loc='upper right', fontsize=10)
    plt.title('Routing Algorithm Comparison\n(with congestion)', fontsize=14)
    plt.savefig('example2_routes.png', dpi=150)
    print("\nSaved: example2_routes.png")

    plt.show()


def example3_congestion_analysis():
    """Example 3: Congestion prediction and analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Congestion Prediction & Analysis")
    print("="*70)

    # Create simulation with prediction enabled
    config = SimulationConfig(
        grid_size=(10, 10),
        spawn_rate=12.0,
        simulation_duration=900,  # 15 minutes
        use_prediction=True,
        prediction_model=PredictionModel.PATTERN_BASED
    )

    sim = TrafficSimulation(config)

    print("\nRunning simulation to collect traffic data...")
    sim.run()

    # Analyze congestion patterns
    print("\nAnalyzing congestion patterns...")

    # Predict for different times of day
    times = {
        'Early Morning (6 AM)': 6.0,
        'Morning Rush (8 AM)': 8.0,
        'Midday (12 PM)': 12.0,
        'Evening Rush (5 PM)': 17.0,
        'Night (10 PM)': 22.0
    }

    predictions = {}
    for label, hour in times.items():
        pred = sim.predictor.predict_network_congestion(
            sim.network, time_of_day=hour, day_of_week=2  # Wednesday
        )
        avg_congestion = sum(pred.values()) / len(pred)
        predictions[label] = avg_congestion
        print(f"{label}: {avg_congestion:.2%}")

    # Visualize predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of predictions
    labels = list(predictions.keys())
    values = list(predictions.values())
    colors_bar = ['#06FFA5' if v < 0.3 else '#FFD166' if v < 0.7 else '#EF476F'
                  for v in values]

    ax1.bar(range(len(labels)), values, color=colors_bar, edgecolor='white', linewidth=2)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Predicted Congestion Level')
    ax1.set_title('Congestion Prediction by Time of Day')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')

    # Congestion heatmap
    visualizer = TrafficVisualizer(sim.network)
    visualizer.ax = ax2
    visualizer.draw_congestion_map()
    ax2.set_title('Actual Network Congestion')

    plt.tight_layout()
    plt.savefig('example3_prediction.png', dpi=150)
    print("\nSaved: example3_prediction.png")

    plt.show()


def example4_vehicle_types():
    """Example 4: Mixed vehicle types simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Mixed Vehicle Types")
    print("="*70)

    # Configure vehicle distribution
    config = SimulationConfig(
        grid_size=(10, 10),
        spawn_rate=10.0,
        simulation_duration=600,
        vehicle_type_distribution={
            VehicleType.CAR: 0.60,
            VehicleType.TRUCK: 0.20,
            VehicleType.BUS: 0.15,
            VehicleType.MOTORCYCLE: 0.03,
            VehicleType.EMERGENCY: 0.02
        }
    )

    sim = TrafficSimulation(config)

    print("\nRunning mixed traffic simulation...")
    results = sim.run()

    # Analyze vehicle types
    vehicle_stats = sim.vehicle_manager.get_stats()
    print("\nVehicle Distribution:")
    for vtype, count in vehicle_stats['vehicles_by_type'].items():
        percentage = (count / results['total_spawned'] * 100) if results['total_spawned'] > 0 else 0
        print(f"  {vtype}: {count} ({percentage:.1f}%)")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart of vehicle types
    types = list(vehicle_stats['vehicles_by_type'].keys())
    counts = list(vehicle_stats['vehicles_by_type'].values())
    colors_pie = ['#06FFA5', '#EF476F', '#FFD166', '#2E86AB', '#FF0000']

    axes[0].pie(counts, labels=types, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
    axes[0].set_title('Vehicle Type Distribution')

    # Statistics over time
    vehicle_counts = {vtype: [] for vtype in VehicleType}
    for stats in sim.stats_history:
        for vtype in VehicleType:
            count = stats['vehicles_by_type'].get(vtype.value, 0)
            vehicle_counts[vtype].append(count)

    time_steps = range(len(sim.stats_history))
    for vtype, color in zip(VehicleType, colors_pie):
        if vtype in vehicle_counts:
            axes[1].plot(time_steps, vehicle_counts[vtype],
                        label=vtype.value, color=color, linewidth=2)

    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Number of Vehicles')
    axes[1].set_title('Vehicle Types Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example4_vehicle_types.png', dpi=150)
    print("\nSaved: example4_vehicle_types.png")

    plt.show()


def example5_network_visualization():
    """Example 5: Network structure visualization"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Network Structure Visualization")
    print("="*70)

    # Create different network sizes
    networks = [
        (6, 6, 'Small Network (6x6)'),
        (10, 10, 'Medium Network (10x10)'),
        (15, 15, 'Large Network (15x15)')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (rows, cols, title) in zip(axes, networks):
        config = SimulationConfig(grid_size=(rows, cols))
        sim = TrafficSimulation(config)

        # Create visualizer for this subplot
        visualizer = TrafficVisualizer(sim.network)
        visualizer.fig = fig
        visualizer.ax = ax

        visualizer.draw_network(show_labels=False)

        network_stats = sim.network.get_network_stats()
        ax.set_title(f'{title}\n{network_stats["num_intersections"]} nodes, '
                    f'{network_stats["num_roads"]} roads')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('example5_networks.png', dpi=150)
    print("\nSaved: example5_networks.png")

    plt.show()


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Simulation", example1_basic_simulation),
        ("Route Comparison", example2_route_comparison),
        ("Congestion Analysis", example3_congestion_analysis),
        ("Vehicle Types", example4_vehicle_types),
        ("Network Visualization", example5_network_visualization)
    ]

    print("\n" + "="*70)
    print("RUNNING ALL TRAFFIC SIMULATOR EXAMPLES")
    print("="*70)

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {name}")
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples_map = {
            '1': example1_basic_simulation,
            '2': example2_route_comparison,
            '3': example3_congestion_analysis,
            '4': example4_vehicle_types,
            '5': example5_network_visualization,
            'all': run_all_examples
        }

        if example_num in examples_map:
            examples_map[example_num]()
        else:
            print("Unknown example number. Use: 1, 2, 3, 4, 5, or 'all'")
    else:
        # Run all by default
        run_all_examples()
