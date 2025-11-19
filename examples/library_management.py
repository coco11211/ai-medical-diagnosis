"""
Style Library Management Example

Demonstrates organizing and managing style models.
"""

import os
from nst_suite.library.manager import StyleLibrary


def main():
    """Run library management examples."""

    print("="*60)
    print("Style Library Management Example")
    print("="*60)

    # Initialize library
    library_dir = "./style_library"
    library = StyleLibrary(library_dir=library_dir)

    print(f"\nLibrary directory: {library_dir}")

    # Example 1: Add models to library
    print("\n" + "="*60)
    print("Example 1: Adding Models to Library")
    print("="*60)

    # Check if models exist to add
    models_to_add = [
        {
            "name": "Starry Night",
            "path": "./checkpoints/starry_night.pth",
            "style_image": "./styles/starry_night.jpg",
            "description": "Van Gogh's iconic Starry Night painting style",
            "tags": ["impressionism", "van_gogh", "classic"]
        },
        {
            "name": "Abstract Waves",
            "path": "./checkpoints/abstract_waves.pth",
            "style_image": "./styles/waves.jpg",
            "description": "Modern abstract wave patterns",
            "tags": ["abstract", "modern", "colorful"]
        },
        {
            "name": "Watercolor",
            "path": "./checkpoints/watercolor.pth",
            "style_image": "./styles/watercolor.jpg",
            "description": "Soft watercolor painting effect",
            "tags": ["watercolor", "soft", "artistic"]
        }
    ]

    # Add models (skip if files don't exist)
    for model_info in models_to_add:
        if os.path.exists(model_info["path"]):
            model_id = library.add_model(
                name=model_info["name"],
                model_path=model_info["path"],
                style_image_path=model_info.get("style_image"),
                description=model_info["description"],
                tags=model_info["tags"],
                copy_files=True
            )
            print(f"Added: {model_info['name']} (ID: {model_id})")
        else:
            print(f"Skipped: {model_info['name']} (file not found)")

    # Example 2: List all models
    print("\n" + "="*60)
    print("Example 2: Listing All Models")
    print("="*60)

    all_models = library.list_models()

    if len(all_models) == 0:
        print("No models in library yet.")
        print("\nTo add models:")
        print("  1. Train a model: nst train style.jpg ./dataset")
        print("  2. Add to library: nst library add 'Style Name' model.pth")
    else:
        print(f"\nFound {len(all_models)} model(s) in library:\n")

        for model in all_models:
            print(f"  ID: {model.id}")
            print(f"  Name: {model.name}")
            print(f"  Description: {model.description}")
            print(f"  Tags: {', '.join(model.tags) if model.tags else 'None'}")
            print(f"  Created: {model.created_at}")
            print()

    # Example 3: Search models by tags
    print("=" * 60)
    print("Example 3: Searching by Tags")
    print("=" * 60)

    if len(all_models) > 0:
        # Get all available tags
        all_tags = set()
        for model in all_models:
            if model.tags:
                all_tags.update(model.tags)

        print(f"\nAvailable tags: {', '.join(sorted(all_tags))}")

        # Search by tag (if any exist)
        if all_tags:
            search_tag = list(all_tags)[0]
            print(f"\nSearching for tag: '{search_tag}'")

            tagged_models = library.list_models(tags=[search_tag])

            print(f"Found {len(tagged_models)} model(s):")
            for model in tagged_models:
                print(f"  - {model.name}")

    # Example 4: Create a collection
    print("\n" + "="*60)
    print("Example 4: Creating Collections")
    print("="*60)

    if len(all_models) >= 2:
        # Create a collection of first 2 models
        model_ids = [m.id for m in all_models[:2]]

        collection = library.create_collection(
            name="My Favorites",
            model_ids=model_ids
        )

        print(f"\nCreated collection: {collection['name']}")
        print(f"Models in collection: {len(collection['models'])}")

    else:
        print("Need at least 2 models to create a collection")

    # Example 5: Export and import models
    print("\n" + "="*60)
    print("Example 5: Export/Import Models")
    print("="*60)

    if len(all_models) > 0:
        # Export first model
        model_to_export = all_models[0]
        export_dir = "./exported_models"

        print(f"\nExporting: {model_to_export.name}")

        library.export_model(
            model_id=model_to_export.id,
            output_dir=export_dir
        )

        print(f"Exported to: {export_dir}")

        # You can import it back:
        # imported_id = library.import_model(export_dir)

    # Example 6: Update metadata
    print("\n" + "="*60)
    print("Example 6: Updating Metadata")
    print("="*60)

    if len(all_models) > 0:
        model_to_update = all_models[0]

        print(f"\nUpdating: {model_to_update.name}")

        library.update_model_metadata(
            model_id=model_to_update.id,
            description="Updated description with more details",
            tags=model_to_update.tags + ["featured"] if model_to_update.tags else ["featured"]
        )

        print("Metadata updated successfully")

    # Example 7: Library statistics
    print("\n" + "="*60)
    print("Example 7: Library Statistics")
    print("="*60)

    stats = library.get_stats()

    print(f"\nLibrary Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Total tags: {stats['total_tags']}")
    print(f"  Library size: {stats['library_size_mb']:.2f} MB")

    if stats['tags']:
        print(f"  All tags: {', '.join(stats['tags'])}")

    # Example 8: Use a model from library
    print("\n" + "="*60)
    print("Example 8: Using Library Model")
    print("="*60)

    if len(all_models) > 0:
        model = all_models[0]

        print(f"\nUsing model: {model.name}")
        print(f"Model path: {library.get_model_path(model.id)}")

        # Example of using it:
        print("\nExample usage:")
        print(f"""
from nst_suite.library import StyleLibrary
from nst_suite import FastStyleTransfer

# Load from library
library = StyleLibrary()
model_path = library.get_model_path('{model.id}')

# Use for transfer
transfer = FastStyleTransfer(model_path)
result = transfer.transfer(content_image)
        """)

    print("\n" + "="*60)
    print("Library Management Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
