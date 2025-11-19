# Traffic Simulator

**Agent-Based Traffic Simulation System for Windows 11**

A comprehensive traffic simulation system featuring agent-based vehicle modeling, ML-powered congestion prediction, advanced route optimization, and real-time visualization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### 1. Agent-Based Vehicle Simulation
- **Autonomous Vehicles**: Each vehicle is an independent agent with its own behavior
- **Multiple Vehicle Types**: Cars, trucks, buses, motorcycles, emergency vehicles
- **Realistic Physics**: Speed, acceleration, and deceleration modeling
- **Dynamic Behavior**: Vehicles respond to traffic conditions in real-time

### 2. Congestion Prediction (ML-Based)
- **Machine Learning Models**: Random Forest and Gradient Boosting
- **Pattern Recognition**: Identifies rush hour patterns and traffic trends
- **Real-Time Predictions**: Forecasts congestion levels for route planning
- **Hotspot Detection**: Automatically identifies congestion-prone areas

### 3. Route Optimization
- **Multiple Algorithms**:
  - **Dijkstra's Algorithm**: Classic shortest path
  - **A* Search**: Heuristic-based pathfinding
  - **Dynamic Routing**: Congestion-aware route selection
  - **Shortest Path**: Pure distance-based routing
- **Alternative Routes**: Find multiple route options
- **Real-Time Adaptation**: Routes update based on current traffic

### 4. Real-Time Visualization
- **Live Simulation Display**: Watch traffic flow in real-time
- **Congestion Heatmaps**: Visual representation of traffic density
- **Route Visualization**: Display and compare different routes
- **Statistical Dashboards**: Comprehensive analytics and metrics
- **Windows 11 Optimized**: Native TkAgg backend for smooth performance

### 5. Comprehensive Analytics
- Travel time analysis
- Speed and congestion metrics
- Vehicle type distribution
- Network utilization statistics
- Performance comparisons

## System Requirements

- **Operating System**: Windows 11 (also compatible with Windows 10, Linux, macOS)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 200MB free space
- **Display**: 1920x1080 or higher recommended for visualization

## Installation

### Step 1: Clone or Download Repository

```bash
git clone <repository-url>
cd ai-medical-diagnosis
```

### Step 2: Install Dependencies

```bash
pip install -r traffic_requirements.txt
```

### Step 3: Verify Installation

```bash
python traffic_main.py --help
```

## Quick Start

### Run Basic Simulation

```bash
python traffic_main.py basic --duration 30 --spawn-rate 10
```

**Options:**
- `--duration`: Simulation duration in minutes (default: 30)
- `--spawn-rate`: Vehicles spawned per minute (default: 5)
- `--grid-rows` / `--grid-cols`: Network grid size (default: 10x10)

### Demo Route Optimization

```bash
python traffic_main.py route_optimization --grid-rows 15 --grid-cols 15
```

Compares different routing algorithms:
- Shortest path by distance
- Dijkstra's algorithm
- A* search
- Dynamic congestion-aware routing

### Demo Congestion Prediction

```bash
python traffic_main.py congestion_prediction --spawn-rate 15
```

Demonstrates ML-based traffic prediction for different times of day.

### Run Example Scripts

```bash
# Run all examples
python examples/traffic_examples.py all

# Run specific example
python examples/traffic_examples.py 1  # Basic simulation
python examples/traffic_examples.py 2  # Route comparison
python examples/traffic_examples.py 3  # Congestion analysis
python examples/traffic_examples.py 4  # Vehicle types
python examples/traffic_examples.py 5  # Network visualization
```

## Usage Examples

### Example 1: Custom Simulation Configuration

```python
from traffic_sim.engine.simulation_engine import TrafficSimulation, SimulationConfig
from traffic_sim.optimization.router import RoutingAlgorithm

# Create custom configuration
config = SimulationConfig(
    time_step=1.0,                    # 1 second per step
    simulation_duration=1800,         # 30 minutes
    grid_size=(12, 12),              # 12x12 grid
    spacing=250.0,                   # 250m between intersections
    spawn_rate=8.0,                  # 8 vehicles per minute
    routing_algorithm=RoutingAlgorithm.A_STAR,
    use_prediction=True,
    visualize=True
)

# Run simulation
sim = TrafficSimulation(config)
results = sim.run()
```

### Example 2: Route Finding

```python
from traffic_sim.optimization.router import Router, RoutingAlgorithm

# Create router
router = Router(network)

# Find optimal route
route = router.find_route(
    origin=0,
    destination=99,
    algorithm=RoutingAlgorithm.DYNAMIC,
    avoid_congestion=True
)

# Get route information
info = router.get_route_info(route)
print(f"Distance: {info['distance_km']:.2f} km")
print(f"Time: {info['time_min']:.2f} minutes")
print(f"Congestion: {info['avg_congestion']:.2%}")
```

### Example 3: Congestion Prediction

```python
from traffic_sim.prediction.congestion_predictor import CongestionPredictor

# Create predictor
predictor = CongestionPredictor()

# Predict congestion
prediction = predictor.predict_network_congestion(
    network,
    time_of_day=8.0,      # 8 AM
    day_of_week=2,        # Wednesday
    weather_factor=1.0
)

# Identify hotspots
hotspots = predictor.get_congestion_hotspots(
    network,
    threshold=0.7,        # 70% congestion
    time_of_day=17.0      # 5 PM
)
```

### Example 4: Visualization

```python
from traffic_sim.visualization.visualizer import TrafficVisualizer

# Create visualizer
visualizer = TrafficVisualizer(network)

# Show congestion heatmap
visualizer.setup_plot()
visualizer.draw_congestion_map()
visualizer.show()

# Save to file
visualizer.save('congestion_map.png', dpi=150)
```

## Command-Line Interface

### Available Scenarios

- `basic`: Standard traffic simulation
- `route_optimization`: Compare routing algorithms
- `congestion_prediction`: Demonstrate ML predictions
- `rush_hour`: High-traffic simulation
- `mixed_traffic`: Various vehicle types

### Common Options

```bash
# Simulation parameters
--duration 60              # 60 minute simulation
--time-step 0.5           # 0.5 second time steps
--spawn-rate 15           # 15 vehicles per minute

# Network configuration
--grid-rows 15            # 15 row grid
--grid-cols 15            # 15 column grid
--spacing 300             # 300m spacing

# Algorithm selection
--algorithm a_star        # Use A* routing
--use-prediction          # Enable ML prediction

# Visualization
--visualize               # Enable visualization
--no-live-viz            # Disable live updates (faster)
--save-viz               # Save visualization to file
--save-stats             # Save statistics plots
```

## Project Structure

```
traffic_sim/
├── agents/              # Vehicle agent system
│   └── vehicle.py      # Vehicle class and manager
├── network/            # Road network system
│   └── road_network.py # Graph-based network
├── optimization/       # Route optimization
│   └── router.py      # Routing algorithms
├── prediction/         # Congestion prediction
│   └── congestion_predictor.py
├── visualization/      # Visualization system
│   └── visualizer.py  # Matplotlib-based viz
├── engine/            # Simulation engine
│   └── simulation_engine.py
├── analytics/         # Analytics and reporting
├── config/           # Configuration
└── scenarios/        # Pre-configured scenarios

examples/
└── traffic_examples.py  # Comprehensive examples

traffic_main.py          # Main entry point
traffic_requirements.txt # Dependencies
TRAFFIC_README.md       # This file
```

## Performance Metrics

The simulator tracks comprehensive metrics:

- **Travel Metrics**: Average/median travel time, distance
- **Speed Metrics**: Average speed, speed distribution
- **Congestion Metrics**: Network congestion level, hotspots
- **Vehicle Metrics**: Completion rate, vehicle distribution
- **Network Metrics**: Road utilization, intersection density

## Algorithm Details

### Route Optimization

1. **Dijkstra's Algorithm**
   - Classic shortest path algorithm
   - Optimal for single-source shortest paths
   - Time complexity: O((V + E) log V)

2. **A* Search**
   - Heuristic-based pathfinding
   - Uses Euclidean distance heuristic
   - Faster than Dijkstra for point-to-point routing
   - Optimal and complete

3. **Dynamic Routing**
   - Considers real-time traffic congestion
   - Applies congestion penalties to edge weights
   - Adapts routes based on current conditions
   - Best for realistic traffic simulation

### Congestion Prediction

- **Random Forest**: Ensemble learning for robust prediction
- **Gradient Boosting**: Sequential ensemble for high accuracy
- **Pattern-Based**: Rule-based prediction for rush hours
- **Features**: Time of day, day of week, historical patterns, weather

## Windows 11 Specific Features

- **Native TkAgg Backend**: Optimized for Windows display
- **Desktop Integration**: Compatible with Windows 11 UI
- **High DPI Support**: Scales properly on 4K displays
- **Performance Optimized**: Efficient rendering on Windows

## Troubleshooting

### Issue: ImportError or ModuleNotFoundError

**Solution:**
```bash
pip install -r traffic_requirements.txt
```

### Issue: Visualization window doesn't appear

**Solution:**
Ensure TkAgg backend is available:
```bash
pip install --upgrade matplotlib
```

On Windows 11, you may need:
```bash
pip install pywin32
```

### Issue: Slow visualization

**Solution:**
- Use `--no-live-viz` for faster simulation
- Reduce grid size: `--grid-rows 8 --grid-cols 8`
- Increase time step: `--time-step 2.0`

### Issue: Out of memory

**Solution:**
- Reduce simulation duration
- Lower spawn rate
- Smaller grid size

## Advanced Configuration

### Custom Vehicle Distribution

```python
from traffic_sim.agents.vehicle import VehicleType

config = SimulationConfig(
    vehicle_type_distribution={
        VehicleType.CAR: 0.65,
        VehicleType.TRUCK: 0.20,
        VehicleType.BUS: 0.10,
        VehicleType.MOTORCYCLE: 0.03,
        VehicleType.EMERGENCY: 0.02
    }
)
```

### Custom Network Creation

```python
from traffic_sim.network.road_network import RoadNetwork

network = RoadNetwork()
network.create_grid_network(
    rows=15,
    cols=15,
    spacing=200.0
)
```

## Performance Tips

1. **Faster Simulation**: Disable live visualization
2. **Better Accuracy**: Smaller time steps (0.5s)
3. **Larger Networks**: Increase grid size (15x15 or 20x20)
4. **Realistic Traffic**: Match spawn rate to network capacity
5. **Save Resources**: Use pattern-based prediction instead of ML

## Future Enhancements

- Real-world map integration (OpenStreetMap)
- Traffic signal optimization
- Multi-modal transportation (pedestrians, bikes)
- Weather effects on traffic
- Accident simulation
- Public transit integration
- Parking simulation
- Deep learning prediction models
- 3D visualization
- Real-time data integration

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

Built with:
- **NetworkX**: Graph algorithms
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **scikit-learn**: Machine learning
- **Python**: Core language

## Support

For issues or questions:
- Check the troubleshooting section
- Review example scripts
- Open an issue on GitHub

## Citation

If you use this simulator in research, please cite:

```
Traffic Simulator - Agent-Based Traffic Simulation System
https://github.com/your-repo/traffic-simulator
```

---

**Happy Simulating!**

For more examples and documentation, see the `examples/` directory and inline code documentation.
