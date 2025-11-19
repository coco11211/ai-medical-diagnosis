"""
Vision-guided control example
Shows how to use camera for object detection and grasping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2
from src.control.robot_controller import RobotController
from src.vision.object_detector import ObjectDetector


def main():
    print("Vision-Guided Robot Control Example")
    print("-" * 40)

    # Create controller with vision enabled
    controller = RobotController(
        enable_vision=True,
        enable_collision_detection=False
    )

    if not controller.initialize():
        print("Failed to initialize vision system")
        return

    # Create object detector
    detector = ObjectDetector(method='color')

    print("\nVision system initialized")
    print("Press 'c' to capture and detect objects")
    print("Press 'g' to grasp detected object")
    print("Press 'q' to quit")

    detected_object = None

    while True:
        # Capture frame
        frame = controller.vision_system.capture_frame()
        if frame is None:
            break

        # Display frame
        display = frame.copy()

        if detected_object is not None:
            # Draw detection
            cv2.rectangle(
                display,
                (detected_object['bbox'][0], detected_object['bbox'][1]),
                (detected_object['bbox'][0] + detected_object['bbox'][2],
                 detected_object['bbox'][1] + detected_object['bbox'][3]),
                (0, 255, 0), 2
            )
            cv2.circle(display, detected_object['centroid'], 5, (0, 0, 255), -1)

        cv2.imshow('Robot Vision', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            # Detect objects (red objects example)
            print("\nDetecting objects...")
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            objects = detector.detect_by_color(frame, lower_red, upper_red, min_area=500)

            if objects:
                detected_object = {
                    'centroid': objects[0].centroid,
                    'bbox': objects[0].bbox
                }
                print(f"Detected object at pixel: {detected_object['centroid']}")
            else:
                print("No objects detected")
                detected_object = None

        elif key == ord('g') and detected_object is not None:
            print("\nPerforming vision-guided grasp...")

            # Convert pixel to 3D (requires depth - using placeholder)
            pixel_coords = np.array(detected_object['centroid'])
            depth = 0.5  # meters (would come from depth camera)

            camera_coords = controller.vision_system.pixel_to_camera_coordinates(
                pixel_coords, depth
            )

            print(f"Target position (camera frame): {camera_coords}")

            # Move to grasp (in real scenario, would transform to robot frame)
            # For now, use camera coordinates as target
            success = controller.move_to_position(camera_coords)

            if success:
                print("Grasp successful!")
            else:
                print("Grasp failed!")

    cv2.destroyAllWindows()
    controller.shutdown()


if __name__ == "__main__":
    main()
