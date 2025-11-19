"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Model configuration."""
    resnet_weights: str = "ResNet50_Weights.IMAGENET1K_V2"
    efficientnet_version: str = "efficientnet-b4"
    num_classes: int = 10
    input_size: tuple = (224, 224)
    batch_size: int = 32
    learning_rate: float = 0.001


class DICOMConfig(BaseModel):
    """DICOM processing configuration."""
    window_center: int = 40
    window_width: int = 400
    target_size: tuple = (512, 512)
    normalize: bool = True


class NLPConfig(BaseModel):
    """NLP configuration."""
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    max_length: int = 512
    embedding_dim: int = 768


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "medical_diagnosis.db"
    username: str = ""
    password: str = ""


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False


class AppConfig(BaseModel):
    """Application configuration."""
    model: ModelConfig = ModelConfig()
    dicom: DICOMConfig = DICOMConfig()
    nlp: NLPConfig = NLPConfig()
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    log_level: str = "INFO"
    device: str = "cuda"  # or "cpu"


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        AppConfig object
    """
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            return AppConfig(**config_dict)
    else:
        # Return default configuration
        return AppConfig()


def save_config(config: AppConfig, config_path: str = "config/config.yaml") -> None:
    """
    Save configuration to YAML file.

    Args:
        config: AppConfig object
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
