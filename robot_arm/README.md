# Robot Arm Controller

A comprehensive robot arm controller system with **inverse kinematics**, **trajectory planning**, **vision integration**, and **collision avoidance** designed for Windows 11.

## Features

### 1. Inverse Kinematics
- **Multiple solving methods**:
  - Numerical optimization (SLSQP)
  - Jacobian-based iterative solver (Damped Least Squares)
  - Cyclic Coordinate Descent (CCD)
- Position and orientation control
- Joint limit enforcement
- Multiple target solving

### 2. Forward Kinematics
- Denavit-Hartenberg (DH) convention
- 6-DOF robot arm (configurable)
- End-effector pose computation
- Jacobian matrix calculation
- Joint position tracking

### 3. Trajectory Planning
- **Interpolation methods**:
  - Linear interpolation
  - Cubic spline
  - Quintic polynomial
  - Trapezoidal velocity profile
- Waypoint-based path planning
- Velocity and acceleration limits
- Smooth trajectory generation
- Path smoothing

### 4. Vision System
- Camera integration (Windows DirectShow)
- Camera calibration (checkerboard method)
- Hand-eye calibration (ArUco markers)
- Pixel to 3D coordinate transformation
- **Object detection**:
  - Color-based detection
  - Contour detection
  - Blob detection
  - Template matching
  - ArUco marker detection
  - Circle detection (Hough transform)
- Vision-guided grasping

### 5. Collision Avoidance
- **Obstacle types**:
  - Spheres
  - Boxes (oriented)
  - Cylinders
- Real-time collision detection
- Trajectory collision checking
- Self-collision detection
- Distance computation
- Repulsive force calculation (potential fields)
- Simple path planning

### 6. Robot Controller
- Integrated control system
- Cartesian and joint space motion
- Pick and place operations
- Trajectory execution
- Emergency stop
- Safety limits enforcement
- State monitoring

### 7. Visualization
- 3D matplotlib visualization
- Robot configuration display
- Trajectory animation
- Obstacle visualization
- Workspace visualization
- Coordinate frame display

## System Requirements

- **Operating System**: Windows 11 (also compatible with Windows 10, Linux, macOS)
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Camera**: Optional (USB webcam for vision features)

## Installation

### Step 1: Clone or Download

```bash
cd robot_arm
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python robot_arm_main.py
```

## Quick Start

### Run Demos

The system includes several demonstration programs:

#### 1. Basic Motion Control
```bash
python robot_arm_main.py basic
```
Demonstrates moving the robot to a target position using inverse kinematics.

#### 2. Trajectory Execution
```bash
python robot_arm_main.py trajectory
```
Shows smooth trajectory planning through multiple waypoints.

#### 3. Collision Avoidance
```bash
python robot_arm_main.py collision
```
Demonstrates collision detection and obstacle avoidance.

#### 4. Pick and Place
```bash
python robot_arm_main.py pick_place
```
Shows a complete pick and place operation.

#### 5. Workspace Visualization
```bash
python robot_arm_main.py workspace
```
Visualizes the robot's reachable workspace.

#### 6. Inverse Kinematics Methods
```bash
python robot_arm_main.py ik
```
Compares different IK solving methods.

#### 7. Interactive Mode
```bash
python robot_arm_main.py interactive
```
Interactive control with commands:
- `move <x> <y> <z>` - Move to position
- `home` - Move to home position
- `obstacle sphere <x> <y> <z> <r>` - Add sphere obstacle
- `obstacle box <x> <y> <z> <sx> <sy> <sz>` - Add box obstacle
- `show` - Visualize current state
- `quit` - Exit

## Usage Examples

### Example 1: Basic Robot Control

```python
from src.control.robot_controller import RobotController
import numpy as np

# Create controller
controller = RobotController()
controller.initialize()

# Move to target position
target = np.array([0.4, 0.2, 0.3])
success = controller.move_to_position(target)

controller.shutdown()
```

### Example 2: Trajectory Planning

```python
from src.control.robot_controller import RobotController
from src.trajectory.path_planner import Waypoint
import numpy as np

controller = RobotController()
controller.initialize()

# Define waypoints
waypoints = [
    Waypoint(position=np.array([0.3, 0.0, 0.2])),
    Waypoint(position=np.array([0.4, 0.2, 0.3])),
    Waypoint(position=np.array([0.3, 0.3, 0.4])),
]

# Execute trajectory
controller.execute_cartesian_path(waypoints)

controller.shutdown()
```

### Example 3: Collision Avoidance

```python
from src.control.robot_controller import RobotController
from src.collision.obstacle import Obstacle
import numpy as np

controller = RobotController(enable_collision_detection=True)
controller.initialize()

# Add obstacle
obstacle = Obstacle.create_sphere(
    center=np.array([0.3, 0.0, 0.3]),
    radius=0.1
)
controller.collision_detector.add_obstacle(obstacle)

# Move with collision checking
target = np.array([0.5, 0.0, 0.2])
success = controller.move_to_position(target, check_collision=True)

controller.shutdown()
```

### Example 4: Pick and Place

```python
from src.control.robot_controller import RobotController
import numpy as np

controller = RobotController()
controller.initialize()

pick_pos = np.array([0.4, 0.2, 0.1])
place_pos = np.array([0.4, -0.2, 0.1])

success = controller.pick_and_place(pick_pos, place_pos)

controller.shutdown()
```

### Example 5: Vision-Guided Grasping

```python
from src.control.robot_controller import RobotController

controller = RobotController(enable_vision=True)
controller.initialize()

# Perform vision-guided grasp
# Detects objects and moves to grasp
success = controller.vision_guided_grasp(
    detect_method='color'
)

controller.shutdown()
```

### Example 6: Visualization

```python
from src.utils.visualizer import RobotVisualizer
from src.kinematics.forward_kinematics import ForwardKinematics
import numpy as np

fk = ForwardKinematics()
viz = RobotVisualizer(fk)

# Plot robot configuration
joint_angles = np.array([0, 0.5, -0.5, 0, 0.5, 0])
viz.plot_robot(joint_angles)

# Visualize workspace
viz.plot_workspace(num_samples=5000)

viz.show()
```

## Project Structure

```
robot_arm/
├── src/
│   ├── kinematics/           # Forward and inverse kinematics
│   │   ├── forward_kinematics.py
│   │   └── inverse_kinematics.py
│   ├── trajectory/           # Path and trajectory planning
│   │   ├── path_planner.py
│   │   └── trajectory_generator.py
│   ├── vision/               # Vision system and object detection
│   │   ├── vision_system.py
│   │   └── object_detector.py
│   ├── collision/            # Collision detection and avoidance
│   │   ├── collision_detector.py
│   │   └── obstacle.py
│   ├── control/              # Main robot controller
│   │   └── robot_controller.py
│   └── utils/                # Visualization utilities
│       └── visualizer.py
├── examples/                 # Example programs
│   ├── example_basic.py
│   ├── example_vision.py
│   └── example_collision.py
├── robot_arm_main.py         # Main program with demos
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Technical Details

### Inverse Kinematics Algorithms

#### 1. Numerical Optimization (Default)
- Uses scipy SLSQP optimizer
- Position + orientation control
- Handles joint limits
- Most accurate but slower

#### 2. Jacobian Method (Damped Least Squares)
- Iterative approach
- Fast convergence
- Good for real-time applications
- May get stuck in local minima

#### 3. Cyclic Coordinate Descent (CCD)
- Simple and fast
- Position-only control
- Good for quick solutions
- Less accurate than optimization

### Trajectory Interpolation

#### Linear Interpolation
- Simplest method
- Discontinuous velocity

#### Cubic Spline
- Smooth trajectory
- Continuous velocity
- Natural boundary conditions

#### Quintic Polynomial
- Zero velocity/acceleration at endpoints
- Very smooth motion
- Good for pick-and-place

#### Trapezoidal Velocity Profile
- Constant acceleration/deceleration
- Predictable timing
- Common in industrial robotics

### DH Parameters

The robot uses standard Denavit-Hartenberg parameters (similar to UR5):
- 6 degrees of freedom
- Revolute joints
- Configurable link lengths and offsets

Default parameters are provided but can be customized for different robot configurations.

## Camera Calibration

### Intrinsic Calibration
1. Print a checkerboard pattern (9x6 recommended)
2. Run calibration:
```python
from src.vision.vision_system import VisionSystem

vision = VisionSystem()
vision.initialize()
vision.calibrate_camera(num_images=20)
vision.save_calibration('camera_calibration.npz')
```

### Hand-Eye Calibration
For camera-to-robot transformation:
1. Attach ArUco marker to fixed position
2. Move robot to multiple poses
3. Run hand-eye calibration

## Performance Tips

### For Real-Time Control
- Use `ik_method='jacobian'` or `'ccd'` for faster IK
- Reduce trajectory points for faster execution
- Use `check_interval` parameter in collision checking
- Disable visualization during execution

### For Accuracy
- Use `ik_method='optimization'` for best accuracy
- Increase number of trajectory points
- Use quintic or cubic spline interpolation
- Lower control frequency for smoother motion

## Troubleshooting

### IK Not Converging
- Try different initial guesses
- Check if target is reachable: `controller.ik.is_reachable(target)`
- Try different IK methods
- Reduce tolerance if needed

### Vision System Not Working
- Check camera ID (try 0, 1, 2)
- Install correct OpenCV: `pip install opencv-python`
- On Windows, ensure DirectShow drivers are available
- Try lowering camera resolution

### Collision Detection Issues
- Verify obstacle positions are correct
- Check safety margin settings
- Visualize obstacles to verify placement
- Adjust link radii if needed

### Visualization Issues
- Update matplotlib: `pip install --upgrade matplotlib`
- Try different matplotlib backend
- Reduce number of workspace samples
- Close previous plot windows

## Advanced Features

### Custom Robot Configuration

Define custom DH parameters:

```python
from src.kinematics.forward_kinematics import ForwardKinematics, DHParameter

custom_dh = [
    DHParameter(a=0, alpha=1.57, d=0.1, theta=0),
    DHParameter(a=0.5, alpha=0, d=0, theta=0),
    DHParameter(a=0.4, alpha=0, d=0, theta=0),
    # ... more joints
]

fk = ForwardKinematics(dh_params=custom_dh)
```

### Custom Trajectory Planning

```python
from src.trajectory.path_planner import PathPlanner, InterpolationType

planner = PathPlanner(
    interpolation_type=InterpolationType.QUINTIC,
    max_velocity=0.5,
    max_acceleration=1.0
)
```

### Custom Obstacle Detection

```python
from src.vision.object_detector import ObjectDetector

detector = ObjectDetector()

# Color-based detection
lower_hsv = np.array([0, 100, 100])  # Red lower bound
upper_hsv = np.array([10, 255, 255])  # Red upper bound
objects = detector.detect_by_color(frame, lower_hsv, upper_hsv)

# Template matching
template = cv2.imread('object_template.png')
objects = detector.detect_by_template(frame, template, threshold=0.8)

# ArUco markers
markers = detector.detect_aruco_markers(frame)
```

## Windows 11 Specific Notes

### Camera Access
- Windows may prompt for camera permissions - allow access
- Use DirectShow backend (automatically configured)
- If camera fails, check Windows Privacy Settings > Camera

### Performance
- Disable Windows visual effects for better performance
- Close unnecessary applications during execution
- Use high-performance power plan

### Dependencies
- All dependencies are compatible with Windows 11
- No compilation required (pure Python + numpy/scipy)
- OpenCV binaries included in pip package

## Future Enhancements

- Real robot hardware interface (serial, USB, Ethernet)
- Reinforcement learning for motion optimization
- Advanced path planning (RRT, RRT*)
- Force/torque control
- Multi-robot coordination
- Gazebo/ROS integration
- Deep learning object detection (YOLO, etc.)
- Real-time trajectory replanning

## License

This project is for educational and research purposes.

## Contributing

Contributions welcome! Areas for improvement:
- Hardware interfaces for specific robots
- Additional IK/trajectory methods
- Better collision avoidance algorithms
- Deep learning integration
- Real-world robot testing

## Support

For issues or questions:
- Check troubleshooting section
- Review example code
- Test with demo programs
- Verify dependencies are installed

## Acknowledgments

Built with:
- **NumPy** & **SciPy** - Scientific computing
- **OpenCV** - Computer vision
- **Matplotlib** - Visualization
- **Python** - Programming language

Inspired by industrial robot controllers (UR5, ABB, KUKA) and research in robotics.

---

**Happy Robot Programming!**

*Developed for Windows 11 with comprehensive inverse kinematics, trajectory planning, vision integration, and collision avoidance.*
