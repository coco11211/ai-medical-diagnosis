"""SHAP-based model explainability."""

import torch
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


class SHAPExplainer:
    """SHAP explainer for medical diagnosis models."""

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: torch.Tensor,
        device: str = 'cuda',
        explainer_type: str = 'deep'
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model to explain
            background_data: Background dataset for SHAP (typically a subset of training data)
            device: Device to run on
            explainer_type: Type of SHAP explainer ('deep', 'gradient', 'kernel')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.explainer_type = explainer_type

        self.model.to(self.device)
        self.model.eval()

        # Initialize SHAP explainer
        logger.info(f"Initializing {explainer_type} SHAP explainer...")

        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(
                model,
                background_data.to(self.device)
            )
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(
                model,
                background_data.to(self.device)
            )
        elif explainer_type == 'kernel':
            # For kernel explainer, we need a prediction function
            def predict_fn(x):
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                    outputs = model(x_tensor)
                    return outputs.cpu().numpy()

            self.explainer = shap.KernelExplainer(
                predict_fn,
                background_data.cpu().numpy()
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

        logger.info("SHAP explainer initialized")

    def explain(
        self,
        input_data: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate SHAP explanations.

        Args:
            input_data: Input data to explain
            class_idx: Specific class to explain (None for all classes)

        Returns:
            SHAP values
        """
        logger.info("Generating SHAP explanations...")

        input_data = input_data.to(self.device)

        # Compute SHAP values
        if self.explainer_type == 'kernel':
            shap_values = self.explainer.shap_values(input_data.cpu().numpy())
        else:
            shap_values = self.explainer.shap_values(input_data)

        # Convert to numpy if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()

        # Select specific class if requested
        if class_idx is not None and isinstance(shap_values, list):
            shap_values = shap_values[class_idx]

        logger.info("SHAP explanations generated")
        return shap_values

    def visualize_image_explanation(
        self,
        input_image: torch.Tensor,
        shap_values: np.ndarray,
        output_path: Optional[str] = None,
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Visualize SHAP explanation for image input.

        Args:
            input_image: Input image
            shap_values: SHAP values
            output_path: Path to save visualization
            class_names: Names of classes
        """
        # Convert image to numpy
        if isinstance(input_image, torch.Tensor):
            image_np = input_image.cpu().numpy()
        else:
            image_np = input_image

        # Handle different image formats
        if len(image_np.shape) == 4:
            image_np = image_np[0]  # Take first image in batch

        # Transpose from (C, H, W) to (H, W, C)
        if image_np.shape[0] in [1, 3]:
            image_np = np.transpose(image_np, (1, 2, 0))

        # Create visualization
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        if image_np.shape[2] == 1:
            plt.imshow(image_np[:, :, 0], cmap='gray')
        else:
            plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')

        # SHAP heatmap
        plt.subplot(1, 3, 2)
        shap_image = np.abs(shap_values).sum(axis=-1) if len(shap_values.shape) > 2 else shap_values
        plt.imshow(shap_image, cmap='hot')
        plt.title('SHAP Importance Heatmap')
        plt.colorbar()
        plt.axis('off')

        # Overlay
        plt.subplot(1, 3, 3)
        if image_np.shape[2] == 1:
            plt.imshow(image_np[:, :, 0], cmap='gray', alpha=0.5)
        else:
            plt.imshow(image_np, alpha=0.5)
        plt.imshow(shap_image, cmap='hot', alpha=0.5)
        plt.title('SHAP Overlay')
        plt.axis('off')

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")

        plt.show()

    def visualize_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        max_display: int = 20
    ) -> None:
        """
        Visualize feature importance using SHAP.

        Args:
            shap_values: SHAP values
            feature_names: Names of features
            output_path: Path to save visualization
            max_display: Maximum number of features to display
        """
        plt.figure(figsize=(10, 8))

        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance saved to {output_path}")

        plt.show()

    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, any]]:
        """
        Get top contributing features.

        Args:
            shap_values: SHAP values
            feature_names: Names of features
            top_k: Number of top features to return

        Returns:
            List of top features with their importance scores
        """
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Flatten if multi-dimensional
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.flatten()

        # Get top indices
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]

        # Prepare results
        top_features = []
        for idx in top_indices:
            feature = {
                'index': int(idx),
                'importance': float(mean_shap[idx])
            }

            if feature_names and idx < len(feature_names):
                feature['name'] = feature_names[idx]

            top_features.append(feature)

        return top_features

    def explain_prediction(
        self,
        input_data: torch.Tensor,
        class_idx: int,
        class_name: str = None
    ) -> Dict[str, any]:
        """
        Complete explanation for a single prediction.

        Args:
            input_data: Input data
            class_idx: Class index to explain
            class_name: Name of the class

        Returns:
            Dictionary with explanation results
        """
        # Get SHAP values
        shap_values = self.explain(input_data, class_idx)

        # Calculate statistics
        mean_importance = np.abs(shap_values).mean()
        max_importance = np.abs(shap_values).max()
        total_positive = (shap_values > 0).sum()
        total_negative = (shap_values < 0).sum()

        explanation = {
            'class_idx': class_idx,
            'class_name': class_name,
            'mean_importance': float(mean_importance),
            'max_importance': float(max_importance),
            'total_positive_contributions': int(total_positive),
            'total_negative_contributions': int(total_negative),
            'shap_values_shape': shap_values.shape
        }

        return explanation
