"""LIME-based model explainability."""

import torch
import numpy as np
from lime import lime_image, lime_text
from lime.wrappers.scikit_image import SegmentationAlgorithm
from typing import Dict, List, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.segmentation import mark_boundaries
from loguru import logger


class LIMEExplainer:
    """LIME explainer for medical diagnosis models."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        device: str = 'cuda'
    ):
        """
        Initialize LIME explainer.

        Args:
            model: Trained model to explain
            class_names: Names of output classes
            device: Device to run on
        """
        self.model = model
        self.class_names = class_names
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.model.eval()

        # Initialize LIME explainers
        self.image_explainer = lime_image.LimeImageExplainer()
        self.text_explainer = lime_text.LimeTextExplainer(class_names=class_names)

        logger.info("LIME explainer initialized")

    def _predict_image(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for image LIME.

        Args:
            images: Batch of images (numpy array)

        Returns:
            Prediction probabilities
        """
        self.model.eval()

        # Convert to tensor
        images_tensor = torch.tensor(images, dtype=torch.float32).to(self.device)

        # Handle shape (LIME gives H, W, C, we need C, H, W)
        if images_tensor.shape[-1] in [1, 3]:
            images_tensor = images_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            outputs = self.model(images_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

    def _predict_text(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for text LIME.

        Args:
            texts: List of text strings

        Returns:
            Prediction probabilities
        """
        # This should be implemented based on your text processing pipeline
        # For now, returning dummy probabilities
        return np.random.rand(len(texts), len(self.class_names))

    def explain_image(
        self,
        image: np.ndarray,
        num_samples: int = 1000,
        num_features: int = 10,
        hide_rest: bool = False
    ) -> Tuple[any, np.ndarray]:
        """
        Generate LIME explanation for image.

        Args:
            image: Input image (H, W, C) in range [0, 1] or [0, 255]
            num_samples: Number of samples for LIME
            num_features: Number of superpixels to show
            hide_rest: Whether to hide other superpixels

        Returns:
            Tuple of (explanation object, explanation mask)
        """
        logger.info("Generating LIME explanation for image...")

        # Ensure image is in correct range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Generate explanation
        explanation = self.image_explainer.explain_instance(
            image,
            self._predict_image,
            top_labels=len(self.class_names),
            hide_color=0,
            num_samples=num_samples
        )

        # Get explanation for top predicted class
        top_label = explanation.top_labels[0]

        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=False,
            num_features=num_features,
            hide_rest=hide_rest
        )

        logger.info("LIME explanation generated")
        return explanation, mask

    def explain_text(
        self,
        text: str,
        num_samples: int = 1000,
        num_features: int = 10
    ) -> any:
        """
        Generate LIME explanation for text.

        Args:
            text: Input text
            num_samples: Number of samples for LIME
            num_features: Number of words to show

        Returns:
            Explanation object
        """
        logger.info("Generating LIME explanation for text...")

        explanation = self.text_explainer.explain_instance(
            text,
            self._predict_text,
            num_features=num_features,
            num_samples=num_samples
        )

        logger.info("LIME explanation generated")
        return explanation

    def visualize_image_explanation(
        self,
        image: np.ndarray,
        explanation: any,
        output_path: Optional[str] = None,
        num_features: int = 10,
        positive_only: bool = False
    ) -> None:
        """
        Visualize LIME explanation for image.

        Args:
            image: Original image
            explanation: LIME explanation object
            output_path: Path to save visualization
            num_features: Number of features to highlight
            positive_only: Show only positive contributions
        """
        # Ensure image is in correct range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Get top label
        top_label = explanation.top_labels[0]

        # Get explanation
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # LIME explanation
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title(f'LIME Explanation\n(Class: {self.class_names[top_label]})')
        axes[1].axis('off')

        # Heatmap
        heatmap_dict = dict(explanation.local_exp[top_label])
        heatmap = np.zeros(image.shape[:2])

        for segment_id, weight in heatmap_dict.items():
            heatmap[mask == segment_id] = weight

        im = axes[2].imshow(heatmap, cmap='RdBu', vmin=-np.abs(heatmap).max(), vmax=np.abs(heatmap).max())
        axes[2].set_title('Feature Importance Heatmap')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")

        plt.show()

    def visualize_text_explanation(
        self,
        explanation: any,
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualize LIME explanation for text.

        Args:
            explanation: LIME explanation object
            output_path: Path to save visualization
        """
        # Get explanation as list
        exp_list = explanation.as_list()

        # Separate positive and negative contributions
        positive = [(word, weight) for word, weight in exp_list if weight > 0]
        negative = [(word, weight) for word, weight in exp_list if weight < 0]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Positive contributions
        if positive:
            words_pos, weights_pos = zip(*positive)
            ax1.barh(range(len(words_pos)), weights_pos, color='green', alpha=0.7)
            ax1.set_yticks(range(len(words_pos)))
            ax1.set_yticklabels(words_pos)
            ax1.set_xlabel('Weight')
            ax1.set_title('Positive Contributions')
            ax1.grid(axis='x', alpha=0.3)

        # Negative contributions
        if negative:
            words_neg, weights_neg = zip(*negative)
            ax2.barh(range(len(words_neg)), weights_neg, color='red', alpha=0.7)
            ax2.set_yticks(range(len(words_neg)))
            ax2.set_yticklabels(words_neg)
            ax2.set_xlabel('Weight')
            ax2.set_title('Negative Contributions')
            ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")

        plt.show()

    def get_feature_importance(
        self,
        explanation: any,
        top_k: int = 10
    ) -> List[Dict[str, any]]:
        """
        Get top contributing features from LIME explanation.

        Args:
            explanation: LIME explanation object
            top_k: Number of top features to return

        Returns:
            List of top features with their importance scores
        """
        # Get top label
        top_label = explanation.top_labels[0]

        # Get feature weights
        feature_weights = explanation.as_list(label=top_label)[:top_k]

        # Format results
        top_features = []
        for feature, weight in feature_weights:
            top_features.append({
                'feature': feature,
                'weight': float(weight),
                'contribution': 'positive' if weight > 0 else 'negative'
            })

        return top_features

    def explain_prediction(
        self,
        image: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        num_samples: int = 1000,
        num_features: int = 10
    ) -> Dict[str, any]:
        """
        Complete explanation for a prediction.

        Args:
            image: Input image (optional)
            text: Input text (optional)
            num_samples: Number of samples for LIME
            num_features: Number of features to show

        Returns:
            Dictionary with explanation results
        """
        result = {}

        if image is not None:
            explanation, mask = self.explain_image(
                image,
                num_samples=num_samples,
                num_features=num_features
            )

            top_label = explanation.top_labels[0]
            result['image_explanation'] = {
                'top_class': self.class_names[top_label],
                'top_class_idx': top_label,
                'top_features': self.get_feature_importance(explanation, num_features)
            }

        if text is not None:
            explanation = self.explain_text(
                text,
                num_samples=num_samples,
                num_features=num_features
            )

            result['text_explanation'] = {
                'top_features': self.get_feature_importance(explanation, num_features)
            }

        return result
