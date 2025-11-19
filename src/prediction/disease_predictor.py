"""Disease prediction engine with confidence scoring."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from loguru import logger


class DiseasePredictor:
    """Comprehensive disease prediction with confidence scores and explanations."""

    def __init__(
        self,
        model: nn.Module,
        disease_names: List[str],
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        top_k: int = 3
    ):
        """
        Initialize disease predictor.

        Args:
            model: Trained prediction model
            disease_names: List of disease names corresponding to class indices
            device: Device to run on
            confidence_threshold: Minimum confidence threshold for predictions
            top_k: Number of top predictions to return
        """
        self.model = model
        self.disease_names = disease_names
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized DiseasePredictor with {len(disease_names)} disease classes")

    def predict(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        return_all_probabilities: bool = False
    ) -> Dict[str, any]:
        """
        Make disease prediction.

        Args:
            image_features: Image feature tensor
            text_features: Text feature tensor
            return_all_probabilities: Whether to return probabilities for all classes

        Returns:
            Dictionary with prediction results
        """
        self.model.eval()

        with torch.no_grad():
            # Move to device
            if image_features is not None:
                image_features = image_features.to(self.device)
            if text_features is not None:
                text_features = text_features.to(self.device)

            # Get model output
            if hasattr(self.model, 'forward') and image_features is not None and text_features is not None:
                # Multi-modal model
                logits = self.model(image_features, text_features)
            elif image_features is not None:
                # Image-only model
                logits = self.model(image_features)
            elif text_features is not None:
                # Text-only model
                logits = self.model(text_features)
            else:
                raise ValueError("At least one of image_features or text_features must be provided")

            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=self.top_k, dim=1)

        # Prepare results
        results = {
            'predictions': [],
            'primary_diagnosis': None,
            'confidence': None,
            'needs_review': False
        }

        # Process batch (assuming batch_size = 1 for simplicity)
        for i in range(top_indices.shape[0]):
            batch_predictions = []

            for j in range(self.top_k):
                disease_idx = top_indices[i, j].item()
                confidence = top_probs[i, j].item()

                prediction = {
                    'disease': self.disease_names[disease_idx],
                    'disease_id': disease_idx,
                    'confidence': confidence,
                    'confidence_level': self._get_confidence_level(confidence)
                }

                batch_predictions.append(prediction)

                # Set primary diagnosis (highest confidence)
                if j == 0:
                    results['primary_diagnosis'] = self.disease_names[disease_idx]
                    results['confidence'] = confidence

                    # Flag for review if confidence is below threshold
                    if confidence < self.confidence_threshold:
                        results['needs_review'] = True

            results['predictions'] = batch_predictions

        # Add all probabilities if requested
        if return_all_probabilities:
            all_probs = probabilities[0].cpu().numpy()
            results['all_probabilities'] = {
                self.disease_names[i]: float(prob)
                for i, prob in enumerate(all_probs)
            }

        return results

    def predict_batch(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> List[Dict[str, any]]:
        """
        Make predictions for a batch.

        Args:
            image_features: Batch of image features
            text_features: Batch of text features

        Returns:
            List of prediction dictionaries
        """
        self.model.eval()

        with torch.no_grad():
            # Move to device
            if image_features is not None:
                image_features = image_features.to(self.device)
                batch_size = image_features.shape[0]
            if text_features is not None:
                text_features = text_features.to(self.device)
                batch_size = text_features.shape[0]

            # Get model output
            if image_features is not None and text_features is not None:
                logits = self.model(image_features, text_features)
            elif image_features is not None:
                logits = self.model(image_features)
            else:
                logits = self.model(text_features)

            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=self.top_k, dim=1)

        # Prepare results for each sample in batch
        batch_results = []

        for i in range(batch_size):
            sample_predictions = []

            for j in range(self.top_k):
                disease_idx = top_indices[i, j].item()
                confidence = top_probs[i, j].item()

                prediction = {
                    'disease': self.disease_names[disease_idx],
                    'disease_id': disease_idx,
                    'confidence': confidence,
                    'confidence_level': self._get_confidence_level(confidence)
                }

                sample_predictions.append(prediction)

            result = {
                'predictions': sample_predictions,
                'primary_diagnosis': self.disease_names[top_indices[i, 0].item()],
                'confidence': top_probs[i, 0].item(),
                'needs_review': top_probs[i, 0].item() < self.confidence_threshold
            }

            batch_results.append(result)

        return batch_results

    def _get_confidence_level(self, confidence: float) -> str:
        """
        Categorize confidence score.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Confidence level string
        """
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"

    def calibrate_confidence(
        self,
        probabilities: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Calibrate confidence scores using temperature scaling.

        Args:
            probabilities: Raw probabilities
            temperature: Temperature parameter

        Returns:
            Calibrated probabilities
        """
        logits = torch.log(probabilities + 1e-10)
        calibrated_logits = logits / temperature
        calibrated_probs = torch.softmax(calibrated_logits, dim=1)

        return calibrated_probs

    def get_uncertainty(
        self,
        probabilities: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate uncertainty metrics.

        Args:
            probabilities: Prediction probabilities

        Returns:
            Dictionary of uncertainty metrics
        """
        # Entropy (higher = more uncertain)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

        # Max probability (lower = more uncertain)
        max_prob = torch.max(probabilities, dim=1)[0]

        # Margin (difference between top 2 predictions, lower = more uncertain)
        top2 = torch.topk(probabilities, k=2, dim=1)[0]
        margin = top2[:, 0] - top2[:, 1]

        return {
            'entropy': entropy.item(),
            'max_probability': max_prob.item(),
            'margin': margin.item(),
            'uncertainty_score': entropy.item() / np.log(len(self.disease_names))  # Normalized
        }

    def save_predictions(
        self,
        predictions: Union[Dict, List[Dict]],
        output_path: str
    ) -> None:
        """
        Save predictions to file.

        Args:
            predictions: Prediction results
            output_path: Path to save predictions
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Predictions saved to {output_path}")

    def load_disease_names(self, path: str) -> None:
        """
        Load disease names from file.

        Args:
            path: Path to disease names file (JSON)
        """
        with open(path, 'r') as f:
            self.disease_names = json.load(f)

        logger.info(f"Loaded {len(self.disease_names)} disease names")

    def get_differential_diagnosis(
        self,
        probabilities: torch.Tensor,
        threshold: float = 0.1
    ) -> List[Dict[str, any]]:
        """
        Get differential diagnosis (all diseases above threshold).

        Args:
            probabilities: Prediction probabilities
            threshold: Minimum probability threshold

        Returns:
            List of differential diagnoses
        """
        differential = []

        for i, prob in enumerate(probabilities[0]):
            if prob.item() >= threshold:
                differential.append({
                    'disease': self.disease_names[i],
                    'probability': prob.item(),
                    'confidence_level': self._get_confidence_level(prob.item())
                })

        # Sort by probability
        differential.sort(key=lambda x: x['probability'], reverse=True)

        return differential


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions."""

    def __init__(
        self,
        models: List[nn.Module],
        disease_names: List[str],
        weights: Optional[List[float]] = None,
        device: str = 'cuda'
    ):
        """
        Initialize ensemble predictor.

        Args:
            models: List of trained models
            disease_names: List of disease names
            weights: Optional weights for each model (must sum to 1)
            device: Device to run on
        """
        self.models = models
        self.disease_names = disease_names
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Set equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights

        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()

        logger.info(f"Initialized EnsemblePredictor with {len(models)} models")

    def predict(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        ensemble_method: str = 'weighted_average'
    ) -> torch.Tensor:
        """
        Make ensemble prediction.

        Args:
            image_features: Image features
            text_features: Text features
            ensemble_method: Method to combine predictions ('weighted_average', 'voting')

        Returns:
            Ensemble probabilities
        """
        all_probabilities = []

        with torch.no_grad():
            for model in self.models:
                # Get prediction from each model
                if image_features is not None and text_features is not None:
                    logits = model(image_features.to(self.device), text_features.to(self.device))
                elif image_features is not None:
                    logits = model(image_features.to(self.device))
                else:
                    logits = model(text_features.to(self.device))

                probabilities = torch.softmax(logits, dim=1)
                all_probabilities.append(probabilities)

        # Stack probabilities
        all_probabilities = torch.stack(all_probabilities)

        # Combine predictions
        if ensemble_method == 'weighted_average':
            weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)
            ensemble_probs = (all_probabilities * weights_tensor).sum(dim=0)
        elif ensemble_method == 'voting':
            # Hard voting
            predictions = torch.argmax(all_probabilities, dim=2)
            ensemble_pred = torch.mode(predictions, dim=0)[0]
            ensemble_probs = torch.zeros_like(all_probabilities[0])
            ensemble_probs.scatter_(1, ensemble_pred.unsqueeze(1), 1.0)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")

        return ensemble_probs
