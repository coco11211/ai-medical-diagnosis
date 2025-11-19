"""Unit tests for NLP modules."""

import pytest
from src.nlp.text_preprocessor import TextPreprocessor
from src.nlp.symptom_analyzer import SymptomAnalyzer, SymptomClassifier
import torch


class TestTextPreprocessor:
    """Test TextPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor(
            remove_stopwords=True,
            lowercase=True,
            lemmatize=True
        )

        assert preprocessor.remove_stopwords is True
        assert preprocessor.lowercase is True
        assert preprocessor.lemmatize is True

    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()
        text = "Check this URL: https://example.com  and  email: test@test.com"

        cleaned = preprocessor.clean_text(text)

        assert "https://example.com" not in cleaned
        assert "test@test.com" not in cleaned
        assert "  " not in cleaned

    def test_tokenize(self, sample_text):
        """Test tokenization."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize(sample_text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

    def test_preprocess(self, sample_text):
        """Test preprocessing pipeline."""
        preprocessor = TextPreprocessor(lowercase=True)
        processed = preprocessor.preprocess(sample_text)

        assert isinstance(processed, str)
        assert len(processed) > 0

    def test_extract_symptoms(self, sample_text):
        """Test symptom extraction."""
        preprocessor = TextPreprocessor()
        symptoms = preprocessor.extract_symptoms(sample_text)

        assert isinstance(symptoms, list)
        # Should extract 'fever', 'cough' from sample text
        assert any(s in ['fever', 'cough'] for s in symptoms)

    def test_extract_body_parts(self):
        """Test body part extraction."""
        preprocessor = TextPreprocessor()
        text = "Pain in chest and abdomen"
        body_parts = preprocessor.extract_body_parts(text)

        assert isinstance(body_parts, list)
        assert 'chest' in body_parts or 'abdomen' in body_parts


class TestSymptomClassifier:
    """Test SymptomClassifier class."""

    def test_initialization(self, disease_names):
        """Test classifier initialization."""
        classifier = SymptomClassifier(
            embedding_dim=768,
            num_classes=len(disease_names),
            hidden_dims=[512, 256]
        )

        assert isinstance(classifier, torch.nn.Module)

    def test_forward_pass(self, disease_names):
        """Test forward pass."""
        classifier = SymptomClassifier(
            embedding_dim=768,
            num_classes=len(disease_names)
        )

        embeddings = torch.randn(2, 768)
        output = classifier(embeddings)

        assert output.shape == (2, len(disease_names))

    def test_predict(self, disease_names):
        """Test prediction method."""
        classifier = SymptomClassifier(
            embedding_dim=768,
            num_classes=len(disease_names)
        )

        embeddings = torch.randn(2, 768)
        predictions, confidences = classifier.predict(embeddings)

        assert predictions.shape == (2,)
        assert confidences.shape == (2,)
        assert torch.all(confidences >= 0) and torch.all(confidences <= 1)
