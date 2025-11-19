"""
Traffic Simulator - Main Entry Point
Windows 11 Compatible Agent-Based Traffic Simulation

Features:
- Agent-based vehicle simulation
- Congestion prediction (ML-based)
- Route optimization (A*, Dijkstra, Dynamic)
- Real-time visualization
- Comprehensive analytics
"""

import argparse
import sys
import matplotlib
matplotlib.use('TkAgg')  # Windows 11 compatible backend

from traffic_sim.engine.simulation_engine import TrafficSimulation, SimulationConfig
from traffic_sim.visualization.visualizer import TrafficVisualizer, VisualizationMode
from traffic_sim.optimization.router import RoutingAlgorithm
from traffic_sim.prediction.congestion_predictor import PredictionModel
from traffic_sim.agents.vehicle import VehicleType


def run_basic_simulation(args):
    """Run a basic traffic simulation"""
    print("\n" + "="*70)
    print("TRAFFIC SIMULATOR - Basic Simulation")
    print("="*70)

    # Create configuration
    config = SimulationConfig(
        time_step=args.time_step,
        simulation_duration=args.duration * 60,  # Convert to seconds
        grid_size=(args.grid_rows, args.grid_cols),
        spacing=args.spacing,
        spawn_rate=args.spawn_rate,
        routing_algorithm=RoutingAlgorithm[args.algorithm.upper()],
        use_prediction=args.use_prediction,
        visualize=args.visualize
    )

    # Create simulation
    sim = TrafficSimulation(config)

    # Run simulation
    if args.visualize and not args.no_live_viz:
        # Run with live visualization
        visualizer = TrafficVisualizer(sim.network)
        visualizer.setup_plot()

        frame_count = [0]

        def update_callback(step, vehicles):
            if step % 10 == 0:  # Update every 10 steps
                visualizer.update_live(vehicles, frame_count[0])
                frame_count[0] += 1
                matplotlib.pyplot.pause(0.01)

        results = sim.run(callback=update_callback)

        if args.save_viz:
            visualizer.save(f"traffic_simulation_{args.grid_rows}x{args.grid_cols}.png")

        visualizer.show()
    else:
        # Run without live visualization
        results = sim.run()

    # Show statistics visualization
    if args.visualize:
        visualizer = TrafficVisualizer(sim.network)
        fig = visualizer.plot_statistics(sim.stats_history)

        if args.save_stats:
            matplotlib.pyplot.savefig(f"traffic_stats_{args.grid_rows}x{args.grid_cols}.png", dpi=150)

        matplotlib.pyplot.show()

    return results


def demo_route_optimization(args):
    """Demonstrate route optimization algorithms"""
    print("\n" + "="*70)
    print("TRAFFIC SIMULATOR - Route Optimization Demo")
    print("="*70)

    # Create network
    config = SimulationConfig(grid_size=(args.grid_rows, args.grid_cols))
    sim = TrafficSimulation(config)

    # Select random origin and destination
    import random
    intersections = list(sim.network.intersections.keys())
    origin = random.choice(intersections)
    destination = random.choice(intersections)

    while origin == destination or sim.network.get_distance(origin, destination) < 500:
        destination = random.choice(intersections)

    print(f"\nFinding routes from {origin} to {destination}")
    print(f"Distance: {sim.network.get_distance(origin, destination):.0f}m")
    print("-" * 70)

    # Compare different algorithms
    algorithms = [
        RoutingAlgorithm.SHORTEST,
        RoutingAlgorithm.DIJKSTRA,
        RoutingAlgorithm.A_STAR,
        RoutingAlgorithm.DYNAMIC
    ]

    routes = {}
    for algo in algorithms:
        route = sim.router.find_route(origin, destination, algorithm=algo)
        if route:
            info = sim.router.get_route_info(route)
            routes[algo] = (route, info)
            print(f"\n{algo.value.upper()}:")
            print(f"  Distance: {info['distance_km']:.2f} km")
            print(f"  Est. Time: {info['time_min']:.2f} min")
            print(f"  Segments: {info['num_segments']}")
            print(f"  Avg Speed: {info['avg_speed_kmh']:.1f} km/h")
            print(f"  Congestion: {info['avg_congestion']:.2%}")

    # Visualize routes
    if args.visualize:
        visualizer = TrafficVisualizer(sim.network)
        visualizer.setup_plot()
        visualizer.draw_network()

        colors = ['#00FF00', '#00FFFF', '#FF00FF', '#FFFF00']
        for i, (algo, (route, info)) in enumerate(routes.items()):
            visualizer.draw_route(route, colors[i], algo.value, linewidth=2)

        matplotlib.pyplot.legend()
        matplotlib.pyplot.title('Route Optimization Comparison')

        if args.save_viz:
            visualizer.save(f"route_comparison_{args.grid_rows}x{args.grid_cols}.png")

        visualizer.show()


def demo_congestion_prediction(args):
    """Demonstrate congestion prediction"""
    print("\n" + "="*70)
    print("TRAFFIC SIMULATOR - Congestion Prediction Demo")
    print("="*70)

    # Create simulation
    config = SimulationConfig(
        grid_size=(args.grid_rows, args.grid_cols),
        spawn_rate=args.spawn_rate,
        use_prediction=True
    )
    sim = TrafficSimulation(config)

    # Run for a short period to generate data
    print("\nGenerating traffic data...")
    sim.run(duration=300)  # 5 minutes

    # Predict congestion
    print("\nPredicting congestion patterns...")

    # Morning rush hour prediction
    morning_prediction = sim.predictor.predict_network_congestion(
        sim.network, time_of_day=8.0, day_of_week=2
    )

    # Evening rush hour prediction
    evening_prediction = sim.predictor.predict_network_congestion(
        sim.network, time_of_day=17.0, day_of_week=2
    )

    # Night time prediction
    night_prediction = sim.predictor.predict_network_congestion(
        sim.network, time_of_day=2.0, day_of_week=2
    )

    print("\nPredicted Average Congestion:")
    print(f"  Morning Rush (8 AM):  {sum(morning_prediction.values())/len(morning_prediction):.2%}")
    print(f"  Evening Rush (5 PM):  {sum(evening_prediction.values())/len(evening_prediction):.2%}")
    print(f"  Night Time (2 AM):    {sum(night_prediction.values())/len(night_prediction):.2%}")

    # Identify hotspots
    morning_hotspots = sim.predictor.get_congestion_hotspots(
        sim.network, threshold=0.7, time_of_day=8.0, day_of_week=2
    )
    print(f"\nPredicted morning rush hotspots: {len(morning_hotspots)} roads")

    # Visualize
    if args.visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        predictions = [morning_prediction, evening_prediction, night_prediction]
        titles = ['Morning Rush (8 AM)', 'Evening Rush (5 PM)', 'Night Time (2 AM)']

        for ax, pred, title in zip(axes, predictions, titles):
            # Create simple congestion visualization
            congestion_values = list(pred.values())
            ax.hist(congestion_values, bins=20, color='#EF476F', edgecolor='white')
            ax.set_xlabel('Congestion Level')
            ax.set_ylabel('Number of Roads')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.axvline(0.7, color='#FFD166', linestyle='--', linewidth=2, label='Hotspot Threshold')
            ax.legend()

        plt.tight_layout()

        if args.save_viz:
            plt.savefig(f"congestion_prediction_{args.grid_rows}x{args.grid_cols}.png", dpi=150)

        plt.show()


def list_scenarios():
    """List available simulation scenarios"""
    print("\n" + "="*70)
    print("AVAILABLE SIMULATION SCENARIOS")
    print("="*70)

    scenarios = {
        'basic': 'Basic traffic simulation with default parameters',
        'route_optimization': 'Compare different routing algorithms',
        'congestion_prediction': 'Demonstrate ML-based congestion prediction',
        'rush_hour': 'Simulate rush hour traffic patterns',
        'mixed_traffic': 'Simulation with various vehicle types'
    }

    for name, description in scenarios.items():
        print(f"\n{name}:")
        print(f"  {description}")
        print(f"  Usage: python traffic_main.py {name}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Traffic Simulator - Agent-based traffic simulation for Windows 11',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic simulation
  python traffic_main.py basic --duration 30 --spawn-rate 10

  # Demo route optimization
  python traffic_main.py route_optimization --grid-rows 15 --grid-cols 15

  # Demo congestion prediction
  python traffic_main.py congestion_prediction --spawn-rate 15

  # List all scenarios
  python traffic_main.py --list-scenarios
        """
    )

    parser.add_argument('scenario', nargs='?', default='basic',
                       choices=['basic', 'route_optimization', 'congestion_prediction',
                               'rush_hour', 'mixed_traffic'],
                       help='Simulation scenario to run')

    # Simulation parameters
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Simulation duration in minutes (default: 30)')
    parser.add_argument('--time-step', type=float, default=1.0,
                       help='Simulation time step in seconds (default: 1.0)')
    parser.add_argument('--spawn-rate', type=float, default=5.0,
                       help='Vehicle spawn rate per minute (default: 5.0)')

    # Network parameters
    parser.add_argument('--grid-rows', type=int, default=10,
                       help='Number of grid rows (default: 10)')
    parser.add_argument('--grid-cols', type=int, default=10,
                       help='Number of grid columns (default: 10)')
    parser.add_argument('--spacing', type=float, default=200.0,
                       help='Grid spacing in meters (default: 200)')

    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default='a_star',
                       choices=['dijkstra', 'a_star', 'dynamic', 'shortest'],
                       help='Routing algorithm (default: a_star)')
    parser.add_argument('--use-prediction', action='store_true', default=True,
                       help='Enable congestion prediction')

    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Enable visualization')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='Disable visualization')
    parser.add_argument('--no-live-viz', action='store_true',
                       help='Disable live visualization during simulation')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization to file')
    parser.add_argument('--save-stats', action='store_true',
                       help='Save statistics plots to file')

    # Other
    parser.add_argument('--list-scenarios', action='store_true',
                       help='List available scenarios')

    args = parser.parse_args()

    # Handle list scenarios
    if args.list_scenarios:
        list_scenarios()
        return

    # Run selected scenario
    try:
        if args.scenario == 'basic':
            run_basic_simulation(args)
        elif args.scenario == 'route_optimization':
            demo_route_optimization(args)
        elif args.scenario == 'congestion_prediction':
            demo_congestion_prediction(args)
        elif args.scenario == 'rush_hour':
            args.spawn_rate = 20.0  # High spawn rate
            run_basic_simulation(args)
        elif args.scenario == 'mixed_traffic':
            run_basic_simulation(args)

        print("\n" + "="*70)
        print("Simulation completed successfully!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
