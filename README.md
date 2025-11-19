# Robotic Vacuum Mapper

A comprehensive SLAM-based robotic vacuum cleaner simulation system for Windows 11 with autonomous mapping, obstacle detection, and intelligent coverage path planning.

## Features

### 1. **SLAM (Simultaneous Localization and Mapping)**
- **Grid-Based Mapping**: Occupancy grid mapping with probabilistic updates
- **Particle Filter Localization**: Monte Carlo Localization (MCL) for accurate pose estimation
- **Real-Time Map Building**: Dynamic map updates as the robot explores
- **Trajectory Tracking**: Complete history of robot movements

### 2. **Obstacle Detection and Mapping**
- **LiDAR Simulation**: 360-degree laser scanning for precise obstacle detection
- **Ultrasonic Sensors**: Short-range collision avoidance with 8-sensor array
- **Bresenham Ray Tracing**: Efficient line-of-sight calculations
- **Multi-Sensor Fusion**: Combined sensor data for robust mapping

### 3. **Coverage Path Planning**
- **Boustrophedon (Lawn Mower)**: Back-and-forth pattern for efficient room coverage
- **Spiral Coverage**: Inward/outward spiral patterns for specific layouts
- **Path Optimization**: Automatic waypoint reduction and smoothing
- **Coverage Metrics**: Real-time coverage percentage calculation

### 4. **Advanced Visualization**
- **Real-Time Mapping**: Live visualization during exploration
- **Interactive Display**: Matplotlib-based visualization for Windows 11
- **Multi-View Analysis**: Side-by-side algorithm comparison
- **Sensor Visualization**: Display LiDAR beams and detection points
- **High-DPI Support**: Optimized for Windows 11 high-resolution displays

### 5. **Environment Simulation**
- **Empty Room**: Basic rectangular room for testing
- **Furniture Layout**: Rooms with randomized obstacles
- **L-Shaped Rooms**: Complex multi-section layouts
- **Maze Environments**: Dense obstacle courses
- **Multi-Room Layouts**: Connected rooms with doorways

### 6. **Particle Filter Localization**
- **1000+ Particles**: High-accuracy position estimation
- **Motion Prediction**: Odometry-based particle movement
- **Sensor Updates**: Likelihood-based weight adjustment
- **Resampling**: Low-variance resampling for efficiency

## System Requirements

- **Operating System**: Windows 11 (also compatible with Windows 10, Linux, macOS)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for larger maps)
- **Storage**: 200MB free space
- **Display**: 1920x1080 or higher recommended

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ai-medical-diagnosis
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.11.0

### Step 3: Verify Installation

```bash
python vacuum_mapper.py --help
```

## Quick Start

### 1. Basic Room Mapping

Map a simple room with furniture:

```bash
python vacuum_mapper.py --environment furniture --steps 500
```

### 2. Complex Environment

Map an L-shaped room:

```bash
python vacuum_mapper.py --environment l_shape --steps 800
```

### 3. Coverage Planning Algorithms

#### Boustrophedon (Lawn Mower) Pattern

```bash
python vacuum_mapper.py --environment furniture --algorithm boustrophedon
```

#### Spiral Pattern

```bash
python vacuum_mapper.py --environment empty --algorithm spiral
```

#### Compare Algorithms

```bash
python vacuum_mapper.py --environment furniture --algorithm compare
```

### 4. Maze Navigation

```bash
python vacuum_mapper.py --environment maze --steps 1000
```

### 5. Multi-Room Mapping

```bash
python vacuum_mapper.py --environment multi_room --steps 1500
```

## Command Line Options

```bash
python vacuum_mapper.py [OPTIONS]
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--environment` | str | furniture | Environment type (empty, furniture, l_shape, maze, multi_room) |
| `--algorithm` | str | boustrophedon | Coverage algorithm (boustrophedon, spiral, compare) |
| `--steps` | int | 500 | Number of exploration steps |
| `--no-visualize` | flag | False | Disable real-time visualization |
| `--save` | str | None | Path to save result visualization |
| `--grid-size` | int | 200 | Grid size (creates square grid) |

### Examples

**Explore empty room and save results:**
```bash
python vacuum_mapper.py --environment empty --steps 400 --save results/empty_room.png
```

**Large grid with no visualization (faster):**
```bash
python vacuum_mapper.py --grid-size 300 --steps 1000 --no-visualize
```

**Quick test with small grid:**
```bash
python vacuum_mapper.py --grid-size 100 --steps 200
```

## Configuration

Edit `vacuum_config.yaml` to customize behavior:

### Robot Parameters

```yaml
robot:
  width: 0.3              # Robot width in meters
  max_speed: 0.3          # Maximum speed
  angular_speed: 1.0      # Turn rate
```

### SLAM Settings

```yaml
slam:
  grid_size: [200, 200]   # Map dimensions
  resolution: 0.05        # 5cm per cell
```

### Sensor Configuration

```yaml
lidar:
  num_beams: 360          # Full circle scanning
  max_range: 5.0          # 5 meter range
  noise_std: 0.02         # 2cm noise

ultrasonic:
  num_sensors: 8          # 8 sensors around robot
  max_range: 0.5          # 50cm range
```

### Coverage Planning

```yaml
coverage:
  algorithm: 'boustrophedon'
  overlap: 0.05           # 5cm overlap between passes
  direction: 'horizontal' # Sweep direction
```

## Project Structure

```
ai-medical-diagnosis/
├── vacuum_mapper/
│   ├── slam/
│   │   ├── grid_slam.py         # Grid-based SLAM implementation
│   │   └── particle_filter.py   # Particle filter localization
│   ├── sensors/
│   │   ├── lidar.py             # LiDAR sensor simulation
│   │   └── ultrasonic.py        # Ultrasonic sensor array
│   ├── coverage/
│   │   ├── boustrophedon.py     # Lawn mower coverage
│   │   └── spiral.py            # Spiral coverage
│   ├── visualization/
│   │   ├── mapper_viz.py        # Static visualization
│   │   └── real_time_viz.py     # Real-time display
│   └── utils/
│       ├── environment.py       # Environment simulation
│       └── logger.py            # Logging utilities
├── vacuum_mapper.py             # Main entry point
├── vacuum_config.yaml           # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## SLAM Algorithm Details

### Grid-Based Mapping

The system uses occupancy grid mapping where each cell represents:

- **0-49**: Free space (lower = more certain)
- **50**: Unknown/unexplored
- **100-199**: Occupied space (higher = more certain)

### Particle Filter

Monte Carlo Localization with:
- **1000 particles** for pose estimation
- **Prediction step**: Motion model with noise
- **Update step**: Sensor likelihood calculation
- **Resampling**: Low-variance resampling when needed

### Sensor Fusion

- **LiDAR**: Long-range (5m) obstacle detection, 360 beams
- **Ultrasonic**: Short-range (0.5m) collision avoidance, 8 sensors
- **Ray Tracing**: Bresenham's algorithm for efficient computation

## Coverage Planning Algorithms

### Boustrophedon (Lawn Mower)

**Advantages:**
- Systematic complete coverage
- Predictable path length
- Optimal for rectangular rooms

**How it works:**
1. Divide space into parallel stripes
2. Sweep back and forth
3. Handle obstacles by splitting regions

### Spiral Coverage

**Advantages:**
- Natural starting from center or edge
- Good for circular/irregular spaces
- Minimal turns in open areas

**How it works:**
1. Start from initial position
2. Spiral outward (or inward)
3. Avoid obstacles dynamically

## Performance Metrics

The system calculates:

- **Exploration Percentage**: Ratio of explored to total free space
- **Coverage Percentage**: Ratio of planned coverage to free space
- **Path Length**: Total waypoints in coverage path
- **Path Optimization**: Waypoint reduction ratio

## Visualization Features

### Real-Time Display

- **Map Building**: See SLAM map grow as robot explores
- **Robot Position**: Live position and orientation
- **Trajectory**: Historical path visualization
- **Sensor Beams**: LiDAR/ultrasonic visualization

### Static Results

- **Occupancy Grid**: Final mapped environment
- **Coverage Path**: Planned cleaning path
- **Algorithm Comparison**: Side-by-side analysis

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Visualization window doesn't appear

**Solution:**
- Check if matplotlib backend is properly configured
- Try: `python -c "import matplotlib; matplotlib.use('TkAgg')"`
- On Windows 11, ensure graphics drivers are updated

### Issue: "Memory Error" with large grids

**Solution:**
- Reduce `--grid-size` (try 100 or 150)
- Reduce `--steps`
- Close other applications

### Issue: Slow performance

**Solution:**
- Use `--no-visualize` flag
- Reduce grid size
- Reduce sensor resolution in config

### Issue: Robot gets stuck in corners

**Solution:**
- Increase turn angles in config
- Reduce collision threshold
- Use different environment type

## Windows 11 Specific Features

- **Native Matplotlib Backend**: Uses TkAgg for optimal Windows performance
- **High-DPI Support**: Automatic scaling for 4K displays
- **Path Handling**: Windows-compatible file paths
- **Performance Optimization**: Tuned for Windows 11 threading model

## Advanced Usage

### Custom Environments

Create custom environments by modifying `Environment` class:

```python
from vacuum_mapper.utils import Environment

env = Environment(size=(300, 300))
env.create_empty_room()
env.add_circular_obstacle(center=(150, 150), radius=30)
```

### Programmatic Access

Use vacuum mapper as a library:

```python
from vacuum_mapper import VacuumMapper
from vacuum_mapper.utils import Environment

# Create mapper
mapper = VacuumMapper(grid_size=(200, 200))

# Create environment
env = Environment(size=(200, 200))
env.create_room_with_furniture(num_obstacles=3)

# Explore
result = mapper.explore_and_map(env, max_steps=500, visualize=True)

# Plan coverage
coverage = mapper.plan_coverage(algorithm='boustrophedon')

# Visualize
mapper.visualize_results(coverage_path=coverage['path'])
```

### Save and Load Maps

```python
# Save map
mapper.slam.save_map('my_map.npy')

# Load map
mapper.slam.load_map('my_map.npy')
```

## Future Enhancements

- [ ] ROS integration for real robot hardware
- [ ] Deep learning-based obstacle classification
- [ ] Multi-floor mapping support
- [ ] Dynamic obstacle handling (moving objects)
- [ ] 3D mapping with depth sensors
- [ ] Cloud-based map sharing
- [ ] Mobile app control interface
- [ ] Voice command integration
- [ ] Reinforcement learning for optimal paths
- [ ] Multi-robot coordination

## Technical Details

### SLAM Implementation

- **Algorithm**: Grid-based occupancy mapping with particle filter
- **Localization**: Monte Carlo Localization (MCL)
- **Mapping**: Probabilistic occupancy grid
- **Sensor Model**: Gaussian noise model
- **Motion Model**: Differential drive kinematics

### Coverage Planning

- **Boustrophedon**: Cell decomposition with back-and-forth sweeping
- **Spiral**: Expanding/contracting spiral from seed point
- **Optimization**: Douglas-Peucker path simplification

### Performance

- **Grid Updates**: O(n) per sensor beam
- **Particle Filter**: O(m*k) where m=particles, k=sensors
- **Path Planning**: O(n²) where n=grid cells
- **Visualization**: 10-20 FPS real-time updates

## Contributing

Contributions welcome! Areas for improvement:

1. Additional coverage algorithms (e.g., spanning tree)
2. Better exploration strategies
3. Multi-robot support
4. Hardware integration examples
5. Performance optimizations

## License

This project is for educational and research purposes.

## Acknowledgments

Built using:
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **SciPy** - Scientific algorithms

Inspired by:
- iRobot Roomba algorithms
- ROS Navigation Stack
- Probabilistic Robotics (Thrun, Burgard, Fox)

## Support

For issues or questions:
1. Check this README
2. Review `vacuum_config.yaml` settings
3. Enable debug logging
4. Open an issue on GitHub

---

**Happy Mapping!**

*Autonomous robotic vacuum cleaning powered by SLAM technology*
