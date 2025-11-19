"""Symptom analysis using medical NLP models."""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional
from .text_preprocessor import TextPreprocessor
from loguru import logger


class SymptomAnalyzer:
    """Analyze symptoms using medical NLP models (BioBERT, etc.)."""

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        max_length: int = 512,
        device: str = 'cuda',
        use_preprocessing: bool = True
    ):
        """
        Initialize symptom analyzer.

        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            device: Device to run model on
            use_preprocessing: Whether to use text preprocessing
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_preprocessing = use_preprocessing

        # Initialize tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Initialize text preprocessor
        if use_preprocessing:
            self.preprocessor = TextPreprocessor(
                remove_stopwords=False,  # Keep all words for medical context
                lowercase=False,  # BioBERT is cased
                remove_punctuation=False,
                lemmatize=False
            )

        logger.info(f"Initialized SymptomAnalyzer with {model_name} on {self.device}")

    def encode_text(
        self,
        text: str,
        return_attention_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text into embeddings.

        Args:
            text: Input text
            return_attention_mask: Whether to return attention mask

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        # Preprocess if enabled
        if self.use_preprocessing:
            text = self.preprocessor.clean_text(text)

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings, None

    def encode_batch(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            Batch of embeddings
        """
        # Preprocess if enabled
        if self.use_preprocessing:
            texts = [self.preprocessor.clean_text(text) for text in texts]

        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptoms from text.

        Args:
            text: Input text

        Returns:
            List of extracted symptoms
        """
        if self.use_preprocessing:
            return self.preprocessor.extract_symptoms(text)
        return []

    def extract_body_parts(self, text: str) -> List[str]:
        """
        Extract body parts from text.

        Args:
            text: Input text

        Returns:
            List of extracted body parts
        """
        if self.use_preprocessing:
            return self.preprocessor.extract_body_parts(text)
        return []

    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        emb1, _ = self.encode_text(text1)
        emb2, _ = self.encode_text(text2)

        # Calculate cosine similarity
        similarity = torch.cosine_similarity(emb1, emb2, dim=1)

        return similarity.item()

    def analyze_symptoms(
        self,
        text: str
    ) -> Dict[str, any]:
        """
        Comprehensive symptom analysis.

        Args:
            text: Input text describing symptoms

        Returns:
            Dictionary with analysis results
        """
        # Get embeddings
        embeddings, _ = self.encode_text(text)

        # Extract symptoms and body parts
        symptoms = self.extract_symptoms(text)
        body_parts = self.extract_body_parts(text)

        # Prepare result
        result = {
            'text': text,
            'embeddings': embeddings.cpu().numpy(),
            'embedding_dim': embeddings.shape[1],
            'symptoms': symptoms,
            'body_parts': body_parts,
            'num_symptoms': len(symptoms),
            'num_body_parts': len(body_parts)
        }

        logger.debug(f"Analyzed symptoms: found {len(symptoms)} symptoms, {len(body_parts)} body parts")

        return result

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.config.hidden_size


class SymptomClassifier(nn.Module):
    """Neural network classifier for symptom-based disease prediction."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_classes: int = 10,
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.3
    ):
        """
        Initialize symptom classifier.

        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of disease classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = embedding_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        logger.info(f"Initialized SymptomClassifier with {embedding_dim}->{hidden_dims}->{num_classes}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Input embeddings

        Returns:
            Class logits
        """
        return self.classifier(embeddings)

    def predict(
        self,
        embeddings: torch.Tensor,
        return_probabilities: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.

        Args:
            embeddings: Input embeddings
            return_probabilities: Whether to return probabilities

        Returns:
            Tuple of (predictions, confidences)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(embeddings)

            if return_probabilities:
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            else:
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(logits, dim=1)[0]

        return predictions, confidences
