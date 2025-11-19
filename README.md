# Neural Style Transfer Suite

A comprehensive neural style transfer toolkit for Windows 11 with real-time video processing, GPU acceleration, and advanced features.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%2011-blue.svg)

## Features

### Core Capabilities
- ✨ **Real-time Video Style Transfer** - Process videos and webcam feeds in real-time
- 🎨 **Multiple Neural Network Models** - VGG19 and ResNet architectures
- 🚀 **GPU Acceleration** - Full CUDA support for NVIDIA GPUs
- 📹 **4K Video Support** - Process high-resolution videos up to 4K
- 🎬 **Webcam Integration** - Live style transfer from your camera
- ⚡ **Batch Processing** - Process multiple images/videos efficiently
- 🎯 **Custom Style Training** - Train your own style transfer models
- 🌈 **Style Interpolation** - Blend multiple styles smoothly
- 📚 **Style Library Management** - Organize and manage your style models
- 💾 **Multiple Export Formats** - MP4, AVI, MOV, MKV, PNG, JPG, etc.

### Advanced Features
- **Optimization-based Transfer** - High-quality Gatys et al. method
- **Fast Feed-forward Transfer** - Real-time processing with trained models
- **Multi-style Models** - Single model handling multiple styles
- **Spatial Style Blending** - Spatially-varying style application
- **Style Morphing** - Smooth transitions between styles
- **Memory Optimization** - Efficient CUDA memory management
- **Half-Precision (FP16)** - Faster processing on modern GPUs

## System Requirements

### Minimum Requirements
- **OS**: Windows 11 (64-bit)
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **Storage**: 2 GB free space

### Recommended Requirements
- **OS**: Windows 11 (64-bit)
- **Python**: 3.10 or higher
- **RAM**: 16 GB
- **GPU**: NVIDIA RTX 3060 or better
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.x
- **Storage**: 10 GB free space

## Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/coco11211/ai-medical-diagnosis.git
cd ai-medical-diagnosis

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### GPU Setup (Windows 11)

#### 1. Install CUDA Toolkit

Download and install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads):

```bash
# Verify CUDA installation
nvcc --version
```

#### 2. Install cuDNN

1. Download cuDNN from [NVIDIA's website](https://developer.nvidia.com/cudnn)
2. Extract and copy files to CUDA installation directory:
   - `bin/` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin\`
   - `include/` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\include\`
   - `lib/` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\lib\x64\`

#### 3. Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Verify Installation

```bash
# Check system info
nst info

# You should see CUDA information if GPU is properly configured
```

## Quick Start

### 1. Apply Style to an Image

```bash
# Using optimization method (high quality)
nst transfer content.jpg style.jpg output.jpg

# Using fast method (real-time)
nst transfer content.jpg style.jpg output.jpg --method fast --model model.pth
```

### 2. Process a Video

```bash
# Standard processing
nst video input.mp4 output.mp4 --model model.pth

# Real-time mode (faster)
nst video input.mp4 output.mp4 --model model.pth --realtime
```

### 3. Webcam Style Transfer

```bash
nst webcam --model model.pth --camera 0 --fps 30
```

### 4. Batch Process Images

```bash
nst batch ./input_images ./output_images --model model.pth --recursive
```

### 5. Train Custom Style Model

```bash
nst train style.jpg ./content_dataset --epochs 2 --batch-size 4
```

## Usage Examples

### Image Style Transfer

#### Optimization-based (Best Quality)

```python
from nst_suite import OptimizationBasedTransfer
from nst_suite.utils import load_image, save_image

# Load images
content = load_image('content.jpg', max_size=512)
style = load_image('style.jpg', max_size=512)

# Initialize transfer
transfer = OptimizationBasedTransfer(
    content_weight=1.0,
    style_weight=1e6
)

# Apply style transfer
stylized = transfer.transfer(
    content, style,
    num_steps=300,
    show_progress=True
)

# Save result
save_image(stylized, 'output.jpg')
```

#### Fast Transfer (Real-time)

```python
from nst_suite import FastStyleTransfer
from nst_suite.utils import load_image, save_image

# Initialize with pre-trained model
transfer = FastStyleTransfer(model_path='model.pth')

# Load and process image
content = load_image('content.jpg')
stylized = transfer.transfer(content)

# Save result
save_image(stylized, 'output.jpg')
```

### Video Processing

```python
from nst_suite.video import VideoStyleTransfer

# Initialize processor
processor = VideoStyleTransfer(
    model_path='model.pth',
    use_half_precision=True
)

# Process video
processor.process_video(
    input_path='input.mp4',
    output_path='output.mp4',
    batch_size=4
)
```

### Webcam Style Transfer

```python
from nst_suite.video import WebcamStyleTransfer

# Initialize webcam processor
webcam = WebcamStyleTransfer(
    model_path='model.pth',
    camera_id=0,
    resize_factor=0.75
)

# Run webcam transfer
webcam.run(
    target_fps=30,
    save_output='webcam_output.mp4'
)
```

### Style Interpolation

```python
from nst_suite.core import StyleInterpolator
from nst_suite.utils import load_image

# Initialize with multiple models
interpolator = StyleInterpolator(
    model_paths=['style1.pth', 'style2.pth', 'style3.pth']
)

# Load content image
content = load_image('content.jpg')

# Interpolate between styles
weights = [0.5, 0.3, 0.2]  # Must sum to 1.0
result = interpolator.interpolate(content, weights)

# Create smooth transition
transition = interpolator.create_transition(
    content,
    style_a_idx=0,
    style_b_idx=1,
    num_steps=30,
    interpolation='ease_in_out'
)
```

### Batch Processing

```python
from nst_suite.batch import BatchProcessor

# Initialize batch processor
processor = BatchProcessor(
    model_path='model.pth',
    num_workers=4
)

# Process directory of images
processor.process_directory(
    input_dir='./images',
    output_dir='./stylized',
    recursive=True,
    batch_size=8
)
```

### Style Library Management

```python
from nst_suite.library import StyleLibrary

# Initialize library
library = StyleLibrary(library_dir='./style_library')

# Add a model
model_id = library.add_model(
    name='Starry Night',
    model_path='starry_night.pth',
    style_image_path='starry_night.jpg',
    description='Van Gogh\'s Starry Night style',
    tags=['impressionism', 'van_gogh']
)

# List all models
models = library.list_models(tags=['impressionism'])

# Use a model from library
model_path = library.get_model_path(model_id)
```

### Training Custom Models

```python
from nst_suite.training import StyleTrainer

# Initialize trainer
trainer = StyleTrainer(
    style_image_path='style.jpg',
    content_weight=1.0,
    style_weight=1e5
)

# Train model
trainer.train(
    content_dir='./coco_dataset',
    num_epochs=2,
    batch_size=4,
    learning_rate=1e-3,
    save_dir='./checkpoints'
)
```

## Command-Line Interface

### Image Transfer

```bash
# Optimization method
nst transfer content.jpg style.jpg output.jpg --steps 500

# Fast method with custom weights
nst transfer content.jpg style.jpg output.jpg \
    --method fast \
    --model model.pth \
    --content-weight 1.0 \
    --style-weight 1e6
```

### Video Processing

```bash
# Standard processing
nst video input.mp4 output.mp4 --model model.pth --batch-size 8

# Real-time with resize
nst video input.mp4 output.mp4 \
    --model model.pth \
    --realtime \
    --resize-factor 0.75
```

### Webcam

```bash
# Basic webcam
nst webcam --model model.pth

# Custom settings with recording
nst webcam \
    --model model.pth \
    --camera 0 \
    --resize-factor 0.75 \
    --fps 30 \
    --save output.mp4
```

### Batch Processing

```bash
# Process directory
nst batch ./input ./output --model model.pth

# Recursive with custom batch size
nst batch ./input ./output \
    --model model.pth \
    --recursive \
    --batch-size 16
```

### Training

```bash
# Train with defaults
nst train style.jpg ./dataset

# Custom training settings
nst train style.jpg ./dataset \
    --epochs 4 \
    --batch-size 8 \
    --lr 0.001 \
    --image-size 256 \
    --style-weight 1e5 \
    --output-dir ./my_models
```

### Library Management

```bash
# Add model to library
nst library add "Starry Night" model.pth \
    --style-image style.jpg \
    --description "Van Gogh style" \
    --tags "impressionism,van_gogh"

# List all models
nst library list

# Search models
nst library list --tags impressionism --search "van gogh"

# Remove model
nst library remove starry_night

# Show statistics
nst library stats
```

### System Information

```bash
# Show CUDA and system info
nst info
```

## Performance Optimization

### GPU Memory Management

```python
from nst_suite.utils.cuda import optimize_cuda_memory, CUDAMemoryTracker

# Optimize CUDA settings
optimize_cuda_memory()

# Track memory usage
with CUDAMemoryTracker("Video Processing"):
    processor.process_video(input_path, output_path)
```

### Batch Size Tuning

```python
from nst_suite.utils.cuda import get_optimal_batch_size

# Estimate optimal batch size
batch_size = get_optimal_batch_size(
    model=transfer.network,
    input_shape=(3, 720, 1280),
    max_memory_usage=0.8
)
```

### Half-Precision Processing

```python
# Use FP16 for faster processing
processor = VideoStyleTransfer(
    model_path='model.pth',
    use_half_precision=True
)
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
processor.process_video(..., batch_size=2)

# Reduce image resolution
transfer = FastStyleTransfer(..., resize_factor=0.5)

# Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Slow Processing

- Enable GPU acceleration (install CUDA)
- Use half-precision mode
- Reduce batch size for better pipelining
- Lower video resolution with `--resize-factor`

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

## Project Structure

```
nst_suite/
├── __init__.py           # Main package initialization
├── cli.py                # Command-line interface
├── models/               # Neural network models
│   ├── base.py          # Base model class
│   ├── vgg.py           # VGG19 models
│   └── resnet.py        # ResNet models
├── core/                 # Core functionality
│   ├── transfer.py      # Transfer implementations
│   └── interpolation.py # Style interpolation
├── training/             # Training pipeline
│   ├── train.py         # Training logic
│   └── dataset.py       # Dataset classes
├── video/                # Video processing
│   ├── processor.py     # Video processors
│   └── webcam.py        # Webcam integration
├── batch/                # Batch processing
│   └── processor.py     # Batch processors
├── library/              # Style library
│   └── manager.py       # Library management
└── utils/                # Utilities
    ├── image.py         # Image processing
    ├── video.py         # Video utilities
    └── cuda.py          # CUDA utilities
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original style transfer: [Gatys et al., 2016](https://arxiv.org/abs/1508.06576)
- Fast style transfer: [Johnson et al., 2016](https://arxiv.org/abs/1603.08155)
- VGG19 architecture: [Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{neural_style_transfer_suite,
  title = {Neural Style Transfer Suite},
  author = {Neural Style Transfer Suite Team},
  year = {2024},
  url = {https://github.com/coco11211/ai-medical-diagnosis}
}
```

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: https://github.com/coco11211/ai-medical-diagnosis/issues
- Documentation: See `docs/` directory

## Version History

- **1.0.0** (2024) - Initial release
  - Real-time video style transfer
  - Multiple neural network models
  - GPU acceleration
  - Webcam integration
  - Batch processing
  - Style library management
  - Comprehensive documentation
