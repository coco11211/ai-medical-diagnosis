"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    """Create sample tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    return "Patient presents with fever, cough, and shortness of breath"


@pytest.fixture
def sample_symptoms():
    """Create sample symptom list."""
    return ["fever", "cough", "fatigue", "chest pain"]


@pytest.fixture
def disease_names():
    """Create sample disease names."""
    return [
        "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer",
        "Pulmonary Edema", "Pleural Effusion", "Cardiomegaly",
        "Nodule", "Mass", "Atelectasis"
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory."""
    return tmp_path


@pytest.fixture
def device():
    """Get device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
