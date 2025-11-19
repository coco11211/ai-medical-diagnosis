# Self-Driving Car Simulator

A comprehensive self-driving car simulator built with PyBullet for Windows 11. Features advanced autonomous driving capabilities including lane detection, obstacle avoidance, path planning, and sensor fusion.

## Features

### 1. Realistic Physics Simulation
- PyBullet physics engine
- Realistic vehicle dynamics
- 3D environment with roads and obstacles
- Real-time collision detection

### 2. Multi-Sensor System
- **Camera**: RGB camera with 640x480 resolution, 60° FOV
- **LIDAR**: 360-ray laser scanner with 50m range
- **GPS**: Position tracking with realistic noise simulation
- **IMU**: Inertial measurement unit for orientation and acceleration

### 3. Advanced Perception
- **Lane Detection**: Computer vision-based lane detection using:
  - Gaussian blur and edge detection
  - Hough line transform
  - Polynomial lane fitting
  - Lane departure calculation
- **Obstacle Detection**: LIDAR-based obstacle identification
- **Environment Mapping**: Real-time occupancy grid generation

### 4. Intelligent Control Systems
- **Obstacle Avoidance**: Artificial potential field method
  - Repulsive forces from obstacles
  - Safe distance maintenance
  - Collision risk assessment
  - Adaptive speed control
- **Lane Following**: Visual servoing for lane keeping
- **Speed Control**: Adaptive cruise control based on obstacles

### 5. Path Planning
- **A* Algorithm**: Grid-based optimal path planning
  - 8-connected grid search
  - Heuristic-based optimization
  - Obstacle-aware routing
- **Potential Field Method**: Real-time reactive planning
  - Attractive forces toward goal
  - Repulsive forces from obstacles
  - Smooth trajectory generation
- **Pure Pursuit**: Path following controller with lookahead

### 6. Sensor Fusion
- **Extended Kalman Filter**: Combines multiple sensor inputs
  - GPS position updates
  - IMU orientation and velocity updates
  - Optimal state estimation
  - Uncertainty quantification
  - Noise filtering

### 7. Operating Modes
- **Autonomous Mode**: Full self-driving capability
- **Manual Mode**: Keyboard control for testing
- **Real-time Mode Switching**: Toggle between modes on-the-fly

### 8. Visualization
- **Camera View**: Live camera feed with lane overlays
- **LIDAR Visualization**: 2D bird's-eye view of surroundings
- **Status Display**: Real-time position, mode, and uncertainty
- **Performance Metrics**: FPS and sensor data display

## System Requirements

### Windows 11 Specific
- **Operating System**: Windows 11 (also compatible with Windows 10)
- **Python**: 3.8 or higher
- **GPU**: Recommended for smooth visualization (OpenGL support)
- **RAM**: 4GB minimum, 8GB recommended
- **Display**: 1920x1080 or higher

### Dependencies
- PyBullet >= 3.2.5
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- Python 3.8+

## Installation

### Step 1: Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

### Step 2: Install Dependencies

```bash
# Navigate to the project directory
cd ai-medical-diagnosis

# Install all requirements
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pybullet; import cv2; import numpy; print('All dependencies installed!')"
```

## Quick Start

### Running the Simulator

```bash
# Run from the project root
python -m self_driving_car.simulator
```

Or:

```bash
# Navigate to the self_driving_car directory
cd self_driving_car
python simulator.py
```

### Controls

When the simulator starts, you'll see:
- PyBullet 3D visualization window
- Camera view window (if enabled)
- LIDAR visualization window (if enabled)

#### Keyboard Controls

| Key | Function |
|-----|----------|
| **Space** | Toggle between Autonomous and Manual mode |
| **↑ Arrow** | Accelerate (Manual mode) |
| **↓ Arrow** | Brake/Reverse (Manual mode) |
| **← Arrow** | Steer left (Manual mode) |
| **→ Arrow** | Steer right (Manual mode) |
| **C** | Toggle camera view |
| **L** | Toggle LIDAR visualization |
| **R** | Reset simulation |
| **Q / ESC** | Quit simulator |

## Architecture

### Project Structure

```
self_driving_car/
├── __init__.py
├── simulator.py              # Main simulation loop
├── README.md                 # This file
│
├── environment/              # Simulation environment
│   ├── __init__.py
│   └── world.py             # PyBullet world setup
│
├── sensors/                  # Sensor implementations
│   ├── __init__.py
│   ├── camera.py            # RGB camera
│   ├── lidar.py             # LIDAR scanner
│   ├── gps.py               # GPS sensor
│   ├── imu.py               # IMU sensor
│   └── car.py               # Car model with sensors
│
├── perception/               # Perception algorithms
│   ├── __init__.py
│   └── lane_detection.py   # Lane detection
│
├── control/                  # Control systems
│   ├── __init__.py
│   └── obstacle_avoidance.py # Obstacle avoidance
│
├── planning/                 # Path planning
│   ├── __init__.py
│   └── path_planner.py      # A* and potential field
│
└── fusion/                   # Sensor fusion
    ├── __init__.py
    └── kalman_filter.py     # Kalman filter
```

### System Flow

```
Sensors → Perception → Planning → Control → Actuators
   ↓                                          ↑
   └──────── Sensor Fusion ──────────────────┘
```

1. **Sensors**: Collect data (camera, LIDAR, GPS, IMU)
2. **Sensor Fusion**: Combine sensor data using Kalman filter
3. **Perception**: Detect lanes and obstacles
4. **Planning**: Generate optimal path
5. **Control**: Calculate steering and throttle
6. **Actuators**: Apply forces to vehicle

## Technical Details

### Lane Detection Algorithm

```python
1. Convert RGB to grayscale
2. Apply Gaussian blur (5x5 kernel)
3. Canny edge detection (50-150 threshold)
4. Region of interest masking
5. Hough line transform
6. Separate left/right lanes by slope
7. Polynomial fitting
8. Calculate lane departure
9. Generate steering command
```

### Obstacle Avoidance

Uses artificial potential fields:
- **Attractive Force**: Pulls toward goal/lane center
- **Repulsive Force**: Pushes away from obstacles
- **Force Combination**: Weighted sum determines steering
- **Speed Adaptation**: Slows down near obstacles

### Sensor Fusion (Kalman Filter)

State vector: `[x, y, vx, vy, heading, angular_velocity]`

Update cycle:
1. **Prediction**: Estimate state using motion model
2. **GPS Update**: Correct position (x, y)
3. **IMU Update**: Correct heading and angular velocity
4. **Velocity Update**: Correct velocity components

### Path Planning

**A* Algorithm:**
- Grid-based search
- 8-connected neighbors
- Euclidean heuristic
- Optimal path guarantee

**Potential Field:**
- Continuous space
- Real-time reactive
- No pre-planning needed
- Local minima possible

## Configuration

### Sensor Parameters

Edit in respective sensor files:

**Camera** (`sensors/camera.py`):
```python
width = 640          # Image width
height = 480         # Image height
fov = 60            # Field of view (degrees)
```

**LIDAR** (`sensors/lidar.py`):
```python
num_rays = 360      # Number of laser beams
max_range = 50      # Maximum range (meters)
```

**GPS** (`sensors/gps.py`):
```python
noise_std = 0.1     # Position noise (meters)
```

**IMU** (`sensors/imu.py`):
```python
accel_noise_std = 0.01   # Accelerometer noise
gyro_noise_std = 0.001   # Gyroscope noise
```

### Control Parameters

**Obstacle Avoidance** (`control/obstacle_avoidance.py`):
```python
safe_distance = 3.0        # Minimum safe distance (m)
influence_distance = 10.0  # Obstacle influence range (m)
```

**Path Following** (`planning/path_planner.py`):
```python
lookahead_distance = 2.0   # Pure pursuit lookahead (m)
```

### Simulation Parameters

In `simulator.py`:
```python
target_speed = 5.0         # Target speed (m/s)
time_step = 1/240.0        # Simulation timestep
```

## Use Cases

### 1. Research and Development
- Test autonomous driving algorithms
- Develop and validate sensor fusion techniques
- Experiment with path planning methods

### 2. Education
- Learn computer vision and robotics
- Understand Kalman filtering
- Study control systems

### 3. Algorithm Testing
- Benchmark obstacle avoidance strategies
- Compare path planning algorithms
- Validate lane detection methods

### 4. Demonstration
- Showcase autonomous vehicle capabilities
- Present to stakeholders
- Create educational videos

## Performance

### Typical Performance (Windows 11, i7, 16GB RAM, GTX 1060):
- **Simulation**: 240 Hz physics update
- **Visualization**: 30+ FPS
- **Lane Detection**: ~20ms per frame
- **LIDAR Scan**: ~5ms (360 rays)
- **Path Planning**: <10ms (A*)

## Troubleshooting

### Issue: PyBullet window doesn't open
**Solution**:
- Check OpenGL support: `python -c "import pybullet as p; p.connect(p.GUI)"`
- Update graphics drivers
- Try DIRECT mode: `simulator = SelfDrivingCarSimulator(gui=False)`

### Issue: OpenCV window freezes
**Solution**:
- Reduce visualization frequency in `simulator.py`
- Disable camera view with 'C' key
- Check antivirus isn't blocking OpenCV

### Issue: Slow performance
**Solution**:
- Close other applications
- Reduce LIDAR rays: `LIDAR(num_rays=180)`
- Lower camera resolution: `Camera(width=320, height=240)`
- Disable visualizations

### Issue: Import errors
**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python -c "import pybullet, cv2, numpy"
```

### Issue: Car doesn't move
**Solution**:
- Check autonomous mode is enabled (press Space)
- Verify no collision at start
- Reset with 'R' key
- Check console for errors

## Extending the Simulator

### Adding New Sensors

1. Create sensor class in `sensors/`:
```python
class NewSensor:
    def __init__(self):
        pass

    def get_reading(self):
        return data
```

2. Add to car in `sensors/car.py`:
```python
self.new_sensor = NewSensor()
```

### Adding New Control Algorithms

1. Create controller in `control/`:
```python
class NewController:
    def calculate_control(self, state):
        return throttle, steering
```

2. Use in `simulator.py`:
```python
controller = NewController()
throttle, steering = controller.calculate_control(state)
```

### Custom Environments

Modify `environment/world.py`:
```python
def _create_custom_obstacles(self):
    # Add your obstacles
    pass
```

## Known Limitations

1. **Simplified Vehicle Dynamics**: Uses basic force-based control instead of detailed tire models
2. **2D Lane Detection**: Assumes flat road surface
3. **Static Obstacles**: Most obstacles are static (some dynamic ones included)
4. **Simple Road Geometry**: Straight road (can be extended to curves)
5. **No Weather Effects**: Clear conditions only

## Future Enhancements

- Curved road support
- Traffic lights and signs
- Multi-vehicle scenarios
- Reinforcement learning integration
- Deep learning-based perception
- ROS integration
- V2V communication
- Weather simulation
- Different road types (highway, urban, etc.)

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes.

## Acknowledgments

- **PyBullet**: Physics simulation
- **OpenCV**: Computer vision
- Built for Windows 11 compatibility

## Support

For issues or questions:
- Check troubleshooting section
- Review code comments
- Create GitHub issue

## Version History

- **v1.0.0** (2024): Initial release
  - PyBullet integration
  - Multi-sensor system
  - Lane detection
  - Obstacle avoidance
  - Path planning
  - Sensor fusion

---

**Happy Autonomous Driving!**

For more information, see the code documentation and comments in each module.
