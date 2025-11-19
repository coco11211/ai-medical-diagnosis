"""
Style Interpolation Example

Demonstrates blending and morphing between multiple styles.
"""

import os
import numpy as np

from nst_suite.core.interpolation import StyleInterpolator, SpatialStyleInterpolator
from nst_suite.utils.image import load_image, save_image


def main():
    """Run style interpolation examples."""

    print("="*60)
    print("Style Interpolation Example")
    print("="*60)

    # Setup paths
    content_path = "content.jpg"
    model_paths = [
        "./checkpoints/style1.pth",
        "./checkpoints/style2.pth",
        "./checkpoints/style3.pth"
    ]

    # Check which models exist
    available_models = [p for p in model_paths if os.path.exists(p)]

    if len(available_models) < 2:
        print("\nError: At least 2 style models required for interpolation")
        print("Train multiple models:")
        print("  nst train style1.jpg ./dataset --output-dir ./checkpoints/style1")
        print("  nst train style2.jpg ./dataset --output-dir ./checkpoints/style2")
        return

    if not os.path.exists(content_path):
        print(f"\nError: Content image not found: {content_path}")
        return

    # Load content image
    print(f"\nLoading content image...")
    content = load_image(content_path, max_size=512)

    output_dir = "./outputs/interpolation"
    os.makedirs(output_dir, exist_ok=True)

    # Example 1: Simple interpolation between two styles
    print("\n" + "="*60)
    print("Example 1: Two-Style Interpolation")
    print("="*60)

    interpolator = StyleInterpolator(
        model_paths=available_models[:2]
    )

    print("Creating interpolation steps...")

    # Create 5 interpolation steps
    for i, alpha in enumerate(np.linspace(0, 1, 5)):
        weights = [1 - alpha, alpha]

        print(f"  Step {i+1}/5: weights = [{weights[0]:.2f}, {weights[1]:.2f}]")

        result = interpolator.interpolate(content, weights)

        output_path = os.path.join(output_dir, f"interpolation_step_{i+1}.jpg")
        save_image(result, output_path)

    print(f"Saved interpolation steps to: {output_dir}")

    # Example 2: Smooth transition video
    print("\n" + "="*60)
    print("Example 2: Smooth Transition Video")
    print("="*60)

    video_output = "./outputs/style_morph.mp4"

    print("Creating morphing video...")
    print("This will create a smooth transition between styles")

    interpolator.create_morphing_video(
        content_image=content,
        output_path=video_output,
        style_sequence=[0, 1, 0],  # Style A -> Style B -> Style A
        steps_per_transition=30,
        fps=30.0
    )

    print(f"Morphing video saved to: {video_output}")

    # Example 3: Three-style blend
    if len(available_models) >= 3:
        print("\n" + "="*60)
        print("Example 3: Three-Style Blend")
        print("="*60)

        interpolator_3 = StyleInterpolator(
            model_paths=available_models[:3]
        )

        # Different weight combinations
        weight_combinations = [
            [1.0, 0.0, 0.0],  # Pure style 1
            [0.0, 1.0, 0.0],  # Pure style 2
            [0.0, 0.0, 1.0],  # Pure style 3
            [0.5, 0.5, 0.0],  # Blend 1 & 2
            [0.5, 0.0, 0.5],  # Blend 1 & 3
            [0.0, 0.5, 0.5],  # Blend 2 & 3
            [0.33, 0.33, 0.34],  # Equal blend
        ]

        for i, weights in enumerate(weight_combinations):
            print(f"  Blend {i+1}: weights = {weights}")

            result = interpolator_3.interpolate(content, np.array(weights))

            output_path = os.path.join(output_dir, f"blend_{i+1}.jpg")
            save_image(result, output_path)

    # Example 4: Spatial interpolation (gradient blend)
    print("\n" + "="*60)
    print("Example 4: Spatial Style Blending")
    print("="*60)

    spatial_interpolator = SpatialStyleInterpolator(
        model_paths=available_models[:2]
    )

    # Horizontal gradient
    print("Creating horizontal gradient blend...")
    result_horizontal = spatial_interpolator.create_gradient_blend(
        content_image=content,
        style_a_idx=0,
        style_b_idx=1,
        direction='horizontal'
    )

    output_path = os.path.join(output_dir, "spatial_horizontal.jpg")
    save_image(result_horizontal, output_path)

    # Vertical gradient
    print("Creating vertical gradient blend...")
    result_vertical = spatial_interpolator.create_gradient_blend(
        content_image=content,
        style_a_idx=0,
        style_b_idx=1,
        direction='vertical'
    )

    output_path = os.path.join(output_dir, "spatial_vertical.jpg")
    save_image(result_vertical, output_path)

    # Radial gradient
    print("Creating radial gradient blend...")
    result_radial = spatial_interpolator.create_gradient_blend(
        content_image=content,
        style_a_idx=0,
        style_b_idx=1,
        direction='radial'
    )

    output_path = os.path.join(output_dir, "spatial_radial.jpg")
    save_image(result_radial, output_path)

    print(f"Spatial blends saved to: {output_dir}")

    print("\n" + "="*60)
    print("Style Interpolation Complete!")
    print("="*60)
    print(f"\nAll results saved in: {output_dir}")


if __name__ == "__main__":
    main()
