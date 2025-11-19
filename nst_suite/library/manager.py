"""Style library management for organizing and accessing style models."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class StyleModel:
    """
    Metadata for a style model.
    """
    id: str
    name: str
    model_path: str
    style_image_path: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None
    model_type: str = "fast"  # "fast" or "optimization"
    architecture: str = "vgg19"  # "vgg19", "resnet", etc.

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'StyleModel':
        """Create from dictionary."""
        return cls(**data)


class StyleLibrary:
    """
    Manage a library of style transfer models.
    """

    def __init__(self, library_dir: str = "./style_library"):
        """
        Initialize style library.

        Args:
            library_dir: Directory to store library data
        """
        self.library_dir = Path(library_dir)
        self.models_dir = self.library_dir / "models"
        self.styles_dir = self.library_dir / "styles"
        self.metadata_file = self.library_dir / "metadata.json"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.styles_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.models: Dict[str, StyleModel] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                for model_id, model_data in data.items():
                    self.models[model_id] = StyleModel.from_dict(model_data)

    def _save_metadata(self):
        """Save metadata to file."""
        data = {
            model_id: model.to_dict()
            for model_id, model in self.models.items()
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_model(
        self,
        name: str,
        model_path: str,
        style_image_path: Optional[str] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        copy_files: bool = True
    ) -> str:
        """
        Add a model to the library.

        Args:
            name: Name of the style
            model_path: Path to model file
            style_image_path: Optional path to style image
            description: Optional description
            author: Optional author name
            tags: Optional list of tags
            copy_files: Whether to copy files to library directory

        Returns:
            Model ID
        """
        # Generate unique ID
        model_id = self._generate_id(name)

        # Copy or reference files
        if copy_files:
            # Copy model file
            model_filename = f"{model_id}.pth"
            dest_model_path = self.models_dir / model_filename
            shutil.copy2(model_path, dest_model_path)
            model_path = str(dest_model_path)

            # Copy style image if provided
            if style_image_path:
                ext = os.path.splitext(style_image_path)[1]
                style_filename = f"{model_id}{ext}"
                dest_style_path = self.styles_dir / style_filename
                shutil.copy2(style_image_path, dest_style_path)
                style_image_path = str(dest_style_path)

        # Create model metadata
        from datetime import datetime
        model = StyleModel(
            id=model_id,
            name=name,
            model_path=model_path,
            style_image_path=style_image_path,
            description=description,
            author=author,
            tags=tags or [],
            created_at=datetime.now().isoformat()
        )

        # Add to library
        self.models[model_id] = model
        self._save_metadata()

        print(f"Added style '{name}' with ID: {model_id}")
        return model_id

    def remove_model(self, model_id: str, delete_files: bool = True):
        """
        Remove a model from the library.

        Args:
            model_id: Model ID to remove
            delete_files: Whether to delete associated files
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in library")

        model = self.models[model_id]

        # Delete files if requested
        if delete_files:
            # Delete model file if it's in the library directory
            model_path = Path(model.model_path)
            if model_path.is_relative_to(self.models_dir):
                model_path.unlink(missing_ok=True)

            # Delete style image if it's in the library directory
            if model.style_image_path:
                style_path = Path(model.style_image_path)
                if style_path.is_relative_to(self.styles_dir):
                    style_path.unlink(missing_ok=True)

        # Remove from library
        del self.models[model_id]
        self._save_metadata()

        print(f"Removed style '{model.name}' (ID: {model_id})")

    def get_model(self, model_id: str) -> StyleModel:
        """
        Get a model by ID.

        Args:
            model_id: Model ID

        Returns:
            StyleModel object
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in library")

        return self.models[model_id]

    def list_models(
        self,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None
    ) -> List[StyleModel]:
        """
        List all models in the library.

        Args:
            tags: Optional list of tags to filter by
            search: Optional search query for name/description

        Returns:
            List of StyleModel objects
        """
        models = list(self.models.values())

        # Filter by tags
        if tags:
            models = [
                model for model in models
                if model.tags and any(tag in model.tags for tag in tags)
            ]

        # Filter by search query
        if search:
            search_lower = search.lower()
            models = [
                model for model in models
                if (search_lower in model.name.lower() or
                    (model.description and search_lower in model.description.lower()))
            ]

        return models

    def get_model_path(self, model_id: str) -> str:
        """
        Get the path to a model file.

        Args:
            model_id: Model ID

        Returns:
            Path to model file
        """
        model = self.get_model(model_id)
        return model.model_path

    def export_model(self, model_id: str, output_dir: str):
        """
        Export a model and its metadata to a directory.

        Args:
            model_id: Model ID to export
            output_dir: Output directory
        """
        model = self.get_model(model_id)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy model file
        model_filename = f"{model.name}.pth"
        shutil.copy2(model.model_path, output_path / model_filename)

        # Copy style image if exists
        if model.style_image_path:
            ext = os.path.splitext(model.style_image_path)[1]
            style_filename = f"{model.name}_style{ext}"
            shutil.copy2(model.style_image_path, output_path / style_filename)

        # Write metadata
        metadata = model.to_dict()
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Exported style '{model.name}' to {output_dir}")

    def import_model(self, import_dir: str) -> str:
        """
        Import a model from a directory.

        Args:
            import_dir: Directory containing model and metadata

        Returns:
            Model ID
        """
        import_path = Path(import_dir)
        metadata_file = import_path / "metadata.json"

        if not metadata_file.exists():
            raise ValueError(f"No metadata.json found in {import_dir}")

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Find model file
        model_files = list(import_path.glob("*.pth"))
        if not model_files:
            raise ValueError(f"No model file (.pth) found in {import_dir}")

        model_path = str(model_files[0])

        # Find style image
        style_image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            style_files = list(import_path.glob(f"*_style{ext}"))
            if style_files:
                style_image_path = str(style_files[0])
                break

        # Add to library
        return self.add_model(
            name=metadata.get('name', 'Imported Style'),
            model_path=model_path,
            style_image_path=style_image_path,
            description=metadata.get('description'),
            author=metadata.get('author'),
            tags=metadata.get('tags'),
            copy_files=True
        )

    def update_model_metadata(
        self,
        model_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Update model metadata.

        Args:
            model_id: Model ID
            name: Optional new name
            description: Optional new description
            tags: Optional new tags
        """
        model = self.get_model(model_id)

        if name is not None:
            model.name = name
        if description is not None:
            model.description = description
        if tags is not None:
            model.tags = tags

        self._save_metadata()
        print(f"Updated metadata for style '{model.name}'")

    def _generate_id(self, name: str) -> str:
        """
        Generate a unique ID for a model.

        Args:
            name: Model name

        Returns:
            Unique ID
        """
        # Create base ID from name
        base_id = name.lower().replace(' ', '_').replace('-', '_')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')

        # Ensure uniqueness
        if base_id not in self.models:
            return base_id

        # Add counter if needed
        counter = 1
        while f"{base_id}_{counter}" in self.models:
            counter += 1

        return f"{base_id}_{counter}"

    def create_collection(
        self,
        name: str,
        model_ids: List[str]
    ) -> dict:
        """
        Create a collection of models.

        Args:
            name: Collection name
            model_ids: List of model IDs

        Returns:
            Collection metadata
        """
        # Validate all model IDs
        for model_id in model_ids:
            if model_id not in self.models:
                raise ValueError(f"Model '{model_id}' not found")

        collection = {
            'name': name,
            'models': model_ids,
            'created_at': __import__('datetime').datetime.now().isoformat()
        }

        # Save collection
        collections_file = self.library_dir / "collections.json"

        if collections_file.exists():
            with open(collections_file, 'r') as f:
                collections = json.load(f)
        else:
            collections = {}

        collection_id = self._generate_id(name)
        collections[collection_id] = collection

        with open(collections_file, 'w') as f:
            json.dump(collections, f, indent=2)

        print(f"Created collection '{name}' with {len(model_ids)} models")
        return collection

    def get_stats(self) -> dict:
        """
        Get library statistics.

        Returns:
            Dictionary with statistics
        """
        all_tags = set()
        for model in self.models.values():
            if model.tags:
                all_tags.update(model.tags)

        return {
            'total_models': len(self.models),
            'total_tags': len(all_tags),
            'tags': sorted(list(all_tags)),
            'library_size_mb': self._get_directory_size() / (1024 * 1024)
        }

    def _get_directory_size(self) -> int:
        """Get total size of library directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.library_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
