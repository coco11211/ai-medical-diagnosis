# API Reference

Complete API reference for the Neural Style Transfer Suite.

## Table of Contents

1. [Core Transfer](#core-transfer)
2. [Models](#models)
3. [Video Processing](#video-processing)
4. [Batch Processing](#batch-processing)
5. [Style Library](#style-library)
6. [Training](#training)
7. [Utilities](#utilities)

---

## Core Transfer

### OptimizationBasedTransfer

Optimization-based style transfer (Gatys et al. 2016).

```python
from nst_suite.core import OptimizationBasedTransfer

transfer = OptimizationBasedTransfer(
    model=None,              # Optional: Custom model
    device=None,             # Optional: torch.device
    content_weight=1.0,      # Content loss weight
    style_weight=1e6,        # Style loss weight
    tv_weight=1e-3          # Total variation loss weight
)

stylized = transfer.transfer(
    content_image,           # numpy array (H, W, C)
    style_image,            # numpy array (H, W, C)
    num_steps=300,          # Optimization steps
    learning_rate=0.03,     # Learning rate
    init_method='content',  # 'content', 'style', or 'random'
    show_progress=True      # Show progress bar
)
```

### FastStyleTransfer

Fast feed-forward style transfer.

```python
from nst_suite.core import FastStyleTransfer

transfer = FastStyleTransfer(
    model_path='model.pth',  # Path to trained model
    device=None              # Optional: torch.device
)

# Transfer single image
stylized = transfer.transfer(content_image)

# Transfer batch of images
stylized_batch = transfer.transfer_batch([img1, img2, img3])

# Save/load model
transfer.save_model('output.pth')
transfer.load_model('input.pth')
```

### StyleInterpolator

Interpolate between multiple styles.

```python
from nst_suite.core import StyleInterpolator

interpolator = StyleInterpolator(
    model_paths=['style1.pth', 'style2.pth', 'style3.pth'],
    device=None
)

# Blend styles with weights
weights = [0.5, 0.3, 0.2]  # Must sum to 1.0
result = interpolator.interpolate(content_image, weights)

# Create transition between two styles
transition = interpolator.create_transition(
    content_image,
    style_a_idx=0,
    style_b_idx=1,
    num_steps=30,
    interpolation='ease_in_out'  # 'linear', 'ease_in', 'ease_out', 'ease_in_out'
)

# Create morphing video
interpolator.create_morphing_video(
    content_image,
    output_path='morph.mp4',
    style_sequence=[0, 1, 2, 0],
    steps_per_transition=30,
    fps=30.0
)
```

---

## Models

### VGG19StyleTransfer

VGG19-based style transfer model.

```python
from nst_suite.models import VGG19StyleTransfer

model = VGG19StyleTransfer(
    device=None,
    content_layers=['conv4_2'],
    style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
)

# Extract features
features = model.extract_features(image_tensor)

# Compute losses
content_loss = model.compute_content_loss(content_features, target_features)
style_loss = model.compute_style_loss(style_features, target_features)
tv_loss = model.compute_total_variation_loss(image_tensor)
```

### ResNetStyleTransfer

ResNet-based style transfer model.

```python
from nst_suite.models import ResNetStyleTransfer

model = ResNetStyleTransfer(
    device=None,
    content_layers=['layer3'],
    style_layers=['layer1', 'layer2', 'layer3', 'layer4'],
    resnet_version='resnet50'  # 'resnet50', 'resnet101', 'resnet152'
)

features = model.extract_features(image_tensor)
```

---

## Video Processing

### VideoStyleTransfer

Process videos with style transfer.

```python
from nst_suite.video import VideoStyleTransfer

processor = VideoStyleTransfer(
    model_path='model.pth',
    device=None,
    use_half_precision=False  # Use FP16 for speed
)

processor.process_video(
    input_path='input.mp4',
    output_path='output.mp4',
    batch_size=4,
    max_frames=None,
    show_progress=True
)
```

### RealtimeVideoProcessor

Real-time video processing with optimizations.

```python
from nst_suite.video import RealtimeVideoProcessor

processor = RealtimeVideoProcessor(
    model_path='model.pth',
    device=None,
    target_fps=30,
    resize_factor=1.0  # < 1 for faster processing
)

# Process single frame
stylized_frame = processor.process_frame(frame)

# Process video file
processor.process_video_file(
    input_path='input.mp4',
    output_path='output.mp4',
    show_preview=False
)
```

### WebcamStyleTransfer

Real-time webcam style transfer.

```python
from nst_suite.video import WebcamStyleTransfer

webcam = WebcamStyleTransfer(
    model_path='model.pth',
    camera_id=0,
    device=None,
    resize_factor=0.75
)

webcam.run(
    window_name='Style Transfer',
    target_fps=30,
    save_output=None,  # Optional: save to video file
    show_fps=True
)
```

---

## Batch Processing

### BatchProcessor

Process multiple images efficiently.

```python
from nst_suite.batch import BatchProcessor

processor = BatchProcessor(
    model_path='model.pth',
    device=None,
    num_workers=4
)

# Process list of images
processor.process_images(
    input_paths=['img1.jpg', 'img2.jpg'],
    output_dir='./output',
    batch_size=8,
    show_progress=True,
    preserve_structure=False
)

# Process entire directory
processor.process_directory(
    input_dir='./images',
    output_dir='./stylized',
    recursive=True,
    extensions=['.jpg', '.png'],
    batch_size=8
)

# Process videos
processor.process_videos(
    input_paths=['video1.mp4', 'video2.mp4'],
    output_dir='./output',
    batch_size=4
)
```

### MultiStyleBatchProcessor

Apply multiple styles to images.

```python
from nst_suite.batch import MultiStyleBatchProcessor

processor = MultiStyleBatchProcessor(
    model_paths=['style1.pth', 'style2.pth', 'style3.pth'],
    device=None
)

# Apply all styles to one image
processor.process_with_all_styles(
    input_path='image.jpg',
    output_dir='./outputs',
    style_names=['Starry Night', 'Waves', 'Abstract']
)

# Create comparison grid
processor.create_style_grid(
    input_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    output_path='grid.jpg',
    grid_size=(3, 3)  # Optional: (rows, cols)
)
```

---

## Style Library

### StyleLibrary

Manage collection of style models.

```python
from nst_suite.library import StyleLibrary

library = StyleLibrary(library_dir='./style_library')

# Add model
model_id = library.add_model(
    name='Starry Night',
    model_path='model.pth',
    style_image_path='style.jpg',
    description='Van Gogh style',
    author='Your Name',
    tags=['impressionism', 'classic'],
    copy_files=True
)

# Get model
model = library.get_model(model_id)
model_path = library.get_model_path(model_id)

# List models
all_models = library.list_models()
filtered = library.list_models(
    tags=['impressionism'],
    search='van gogh'
)

# Remove model
library.remove_model(model_id, delete_files=True)

# Update metadata
library.update_model_metadata(
    model_id,
    name='New Name',
    description='New description',
    tags=['tag1', 'tag2']
)

# Export/Import
library.export_model(model_id, './export_dir')
imported_id = library.import_model('./import_dir')

# Create collection
collection = library.create_collection(
    name='My Favorites',
    model_ids=[id1, id2, id3]
)

# Get statistics
stats = library.get_stats()
```

---

## Training

### StyleTrainer

Train custom style transfer models.

```python
from nst_suite.training import StyleTrainer

trainer = StyleTrainer(
    style_image_path='style.jpg',
    content_weight=1.0,
    style_weight=1e5,
    tv_weight=1e-6,
    device=None
)

trainer.train(
    content_dir='./coco_dataset',
    num_epochs=2,
    batch_size=4,
    learning_rate=1e-3,
    image_size=256,
    save_dir='./checkpoints',
    save_interval=500
)

# Save/load checkpoints
trainer.save_checkpoint('checkpoint.pth', epoch, iteration)
trainer.load_checkpoint('checkpoint.pth')
```

### StyleDataset

Dataset for training.

```python
from nst_suite.training import StyleDataset

dataset = StyleDataset(
    content_dir='./images',
    image_size=256,
    transform=None  # Optional custom transform
)

# Use with DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
```

---

## Utilities

### Image Processing

```python
from nst_suite.utils import (
    load_image,
    save_image,
    preprocess_image,
    postprocess_image
)

# Load image
image = load_image(
    'image.jpg',
    max_size=512,
    keep_aspect_ratio=True
)

# Save image
save_image(image, 'output.jpg')

# Preprocess for network
tensor = preprocess_image(image, device)

# Postprocess from network
image = postprocess_image(tensor)
```

### Video Processing

```python
from nst_suite.utils import VideoProcessor, WebcamProcessor

# Video processor
video_proc = VideoProcessor()

# Get video info
info = video_proc.get_video_info('video.mp4')

# Read frames
for frame in video_proc.read_video('video.mp4', max_frames=100):
    # Process frame
    pass

# Write video
video_proc.write_video(
    frames,
    'output.mp4',
    fps=30.0,
    fourcc='mp4v'
)

# Extract frames
video_proc.extract_frames(
    'video.mp4',
    './frames',
    interval=1,
    max_frames=None
)

# Webcam processor
with WebcamProcessor(camera_id=0) as webcam:
    frame = webcam.read_frame()
```

### CUDA Utilities

```python
from nst_suite.utils import (
    check_cuda,
    get_cuda_info,
    print_cuda_info,
    optimize_cuda_memory,
    CUDAMemoryTracker
)

# Check CUDA
if check_cuda():
    print("CUDA is available")

# Get info
info = get_cuda_info()
print_cuda_info()

# Optimize
optimize_cuda_memory()

# Track memory
with CUDAMemoryTracker("My Operation"):
    # Your code here
    pass
```

---

## Error Handling

All functions may raise the following exceptions:

- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing files
- `RuntimeError`: CUDA/processing errors
- `OSError`: File I/O errors

Example error handling:

```python
try:
    transfer = FastStyleTransfer(model_path='model.pth')
    result = transfer.transfer(content_image)
except FileNotFoundError:
    print("Model file not found")
except RuntimeError as e:
    print(f"Processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Type Hints

The library uses type hints throughout:

```python
import numpy as np
import torch
from typing import List, Optional, Dict

def transfer(
    content_image: np.ndarray,
    style_image: np.ndarray,
    num_steps: int = 300
) -> np.ndarray:
    pass
```

---

## Configuration

### Device Selection

```python
import torch

# Auto-select
device = None  # Will use CUDA if available

# Specific GPU
device = torch.device('cuda:0')

# CPU only
device = torch.device('cpu')
```

### Memory Management

```python
# Reduce batch size
processor.process_video(..., batch_size=2)

# Use half precision
processor = VideoStyleTransfer(..., use_half_precision=True)

# Resize images
transfer = WebcamStyleTransfer(..., resize_factor=0.5)
```

---

## Best Practices

1. **Always use context managers** for webcam and file operations
2. **Enable GPU** for significantly better performance
3. **Use appropriate batch sizes** based on GPU memory
4. **Save checkpoints** during long training runs
5. **Validate inputs** before processing
6. **Handle errors gracefully** in production code

---

For more examples and tutorials, see the [examples/](../examples/) directory.
