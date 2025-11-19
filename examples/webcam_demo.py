"""
Webcam Style Transfer Demo

Real-time style transfer from webcam with interactive controls.
"""

from nst_suite.video.webcam import WebcamStyleTransfer
import os


def main():
    """Run webcam style transfer demo."""

    print("="*60)
    print("Webcam Style Transfer Demo")
    print("="*60)

    # Setup
    model_path = "./checkpoints/final_model.pth"

    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train a model first:")
        print("  nst train style.jpg ./dataset")
        return

    print("\nSettings:")
    print("  Model: Using trained style transfer model")
    print("  Camera: Default camera (0)")
    print("  Resize Factor: 0.75 (for better performance)")
    print("  Target FPS: 30")

    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Close the window to stop")

    print("\nStarting webcam...")

    # Initialize webcam processor
    webcam = WebcamStyleTransfer(
        model_path=model_path,
        camera_id=0,
        resize_factor=0.75
    )

    # Run webcam style transfer
    try:
        webcam.run(
            window_name='Webcam Style Transfer - Press Q to Quit',
            target_fps=30,
            save_output=None,  # Set to path to record
            show_fps=True
        )

    except KeyboardInterrupt:
        print("\nStopped by user")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Check if webcam is connected")
        print("  - Close other applications using the webcam")
        print("  - Try different camera ID: --camera 1")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
