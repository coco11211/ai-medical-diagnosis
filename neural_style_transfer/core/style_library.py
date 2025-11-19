"""
Style Library Manager
Manages collection of pre-trained styles and models
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
from PIL import Image
import requests
from tqdm import tqdm

from ..utils.image_utils import ImageProcessor


class StyleLibrary:
    """
    Manages a library of style models and style images
    """

    def __init__(self, library_path: str = './style_library'):
        """
        Initialize style library

        Args:
            library_path: Path to library directory
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.library_path / 'models'
        self.styles_dir = self.library_path / 'styles'
        self.previews_dir = self.library_path / 'previews'

        self.models_dir.mkdir(exist_ok=True)
        self.styles_dir.mkdir(exist_ok=True)
        self.previews_dir.mkdir(exist_ok=True)

        # Load or create index
        self.index_path = self.library_path / 'index.json'
        self.index = self._load_index()

        logging.info(f"Style library initialized: {self.library_path}")

    def _load_index(self) -> Dict:
        """Load library index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        else:
            return {'styles': {}, 'version': '1.0'}

    def _save_index(self):
        """Save library index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_style(self,
                 style_id: str,
                 style_image_path: str,
                 model_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 author: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> Dict:
        """
        Add a new style to the library

        Args:
            style_id: Unique identifier for the style
            style_image_path: Path to style image
            model_path: Optional path to trained model
            name: Display name
            description: Style description
            author: Style author
            tags: List of tags

        Returns:
            Style metadata
        """
        if style_id in self.index['styles']:
            logging.warning(f"Style {style_id} already exists. Overwriting.")

        # Copy style image
        style_dest = self.styles_dir / f"{style_id}.jpg"
        shutil.copy(style_image_path, style_dest)

        # Copy model if provided
        model_dest = None
        if model_path:
            model_dest = self.models_dir / f"{style_id}.pth"
            shutil.copy(model_path, model_dest)

        # Create preview
        preview_dest = self.previews_dir / f"{style_id}_preview.jpg"
        self._create_preview(style_image_path, preview_dest)

        # Create metadata
        metadata = {
            'id': style_id,
            'name': name or style_id,
            'description': description or '',
            'author': author or 'Unknown',
            'tags': tags or [],
            'style_image': str(style_dest),
            'model': str(model_dest) if model_dest else None,
            'preview': str(preview_dest)
        }

        # Add to index
        self.index['styles'][style_id] = metadata
        self._save_index()

        logging.info(f"Added style: {style_id}")

        return metadata

    def get_style(self, style_id: str) -> Optional[Dict]:
        """
        Get style metadata

        Args:
            style_id: Style identifier

        Returns:
            Style metadata or None
        """
        return self.index['styles'].get(style_id)

    def list_styles(self, tags: Optional[List[str]] = None) -> List[Dict]:
        """
        List all styles, optionally filtered by tags

        Args:
            tags: Optional list of tags to filter by

        Returns:
            List of style metadata
        """
        styles = list(self.index['styles'].values())

        if tags:
            styles = [
                s for s in styles
                if any(tag in s.get('tags', []) for tag in tags)
            ]

        return styles

    def remove_style(self, style_id: str):
        """Remove a style from the library"""
        if style_id not in self.index['styles']:
            logging.warning(f"Style {style_id} not found")
            return

        metadata = self.index['styles'][style_id]

        # Remove files
        for key in ['style_image', 'model', 'preview']:
            file_path = metadata.get(key)
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()

        # Remove from index
        del self.index['styles'][style_id]
        self._save_index()

        logging.info(f"Removed style: {style_id}")

    def _create_preview(self, style_image_path: str, preview_path: str, size: int = 256):
        """Create thumbnail preview"""
        processor = ImageProcessor()
        image = processor.load_image(style_image_path)
        image = processor.resize_image(image, size)
        processor.save_image(image, preview_path, quality=85)

    def export_library(self, export_path: str):
        """Export entire library to a zip file"""
        import zipfile

        export_path = Path(export_path)

        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.library_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.library_path)
                    zipf.write(file_path, arcname)

        logging.info(f"Library exported to {export_path}")

    def import_library(self, import_path: str):
        """Import library from a zip file"""
        import zipfile

        import_path = Path(import_path)

        with zipfile.ZipFile(import_path, 'r') as zipf:
            zipf.extractall(self.library_path)

        # Reload index
        self.index = self._load_index()

        logging.info(f"Library imported from {import_path}")

    def search_styles(self, query: str) -> List[Dict]:
        """
        Search styles by name, description, or tags

        Args:
            query: Search query

        Returns:
            List of matching styles
        """
        query = query.lower()
        results = []

        for style in self.index['styles'].values():
            # Search in name, description, and tags
            if (query in style.get('name', '').lower() or
                query in style.get('description', '').lower() or
                any(query in tag.lower() for tag in style.get('tags', []))):
                results.append(style)

        return results


class PresetStyles:
    """
    Pre-configured famous art styles
    """

    PRESETS = {
        'starry_night': {
            'name': 'Starry Night',
            'description': 'Vincent van Gogh\'s iconic swirling night sky',
            'tags': ['van_gogh', 'post_impressionism', 'classic'],
            'author': 'Vincent van Gogh'
        },
        'the_scream': {
            'name': 'The Scream',
            'description': 'Edvard Munch\'s expressionist masterpiece',
            'tags': ['munch', 'expressionism', 'classic'],
            'author': 'Edvard Munch'
        },
        'the_wave': {
            'name': 'The Great Wave',
            'description': 'Hokusai\'s famous wave woodblock print',
            'tags': ['hokusai', 'japanese', 'ukiyo-e'],
            'author': 'Katsushika Hokusai'
        },
        'picasso_abstract': {
            'name': 'Picasso Abstract',
            'description': 'Cubist style inspired by Pablo Picasso',
            'tags': ['picasso', 'cubism', 'modern'],
            'author': 'Pablo Picasso'
        },
        'mosaic': {
            'name': 'Mosaic',
            'description': 'Ancient mosaic tile pattern style',
            'tags': ['mosaic', 'geometric', 'ancient'],
            'author': 'Various'
        },
        'watercolor': {
            'name': 'Watercolor',
            'description': 'Soft watercolor painting style',
            'tags': ['watercolor', 'painting', 'soft'],
            'author': 'Various'
        },
        'candy': {
            'name': 'Candy',
            'description': 'Vibrant, colorful candy-like style',
            'tags': ['candy', 'vibrant', 'modern'],
            'author': 'Various'
        },
        'pencil_sketch': {
            'name': 'Pencil Sketch',
            'description': 'Hand-drawn pencil sketch effect',
            'tags': ['sketch', 'drawing', 'black_white'],
            'author': 'Various'
        },
        'ukiyo-e': {
            'name': 'Ukiyo-e',
            'description': 'Traditional Japanese woodblock print style',
            'tags': ['japanese', 'ukiyo-e', 'traditional'],
            'author': 'Various'
        },
        'oil_painting': {
            'name': 'Oil Painting',
            'description': 'Classic oil painting texture and style',
            'tags': ['oil', 'painting', 'classic'],
            'author': 'Various'
        }
    }

    @classmethod
    def get_preset_info(cls, preset_id: str) -> Optional[Dict]:
        """Get information about a preset style"""
        return cls.PRESETS.get(preset_id)

    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available preset IDs"""
        return list(cls.PRESETS.keys())
