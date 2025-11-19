# Quick Start Guide

Get started with Neural Style Transfer Suite in 5 minutes!

## 1. Installation (Windows 11)

```bash
# Clone the repository
git clone https://github.com/coco11211/ai-medical-diagnosis.git
cd ai-medical-diagnosis

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Verify installation
nst info
```

## 2. Your First Style Transfer

### Option A: Optimization-Based (No Training Required)

```bash
# Download test images
# Place your content.jpg and style.jpg in the current directory

# Apply style transfer
nst transfer content.jpg style.jpg output.jpg --steps 300

# Result will be saved as output.jpg
```

### Option B: Fast Transfer (Requires Training)

```bash
# 1. Download COCO dataset or use your own image collection
# 2. Train a model (takes ~30 minutes on GPU)
nst train style.jpg ./path/to/dataset --epochs 2 --batch-size 4

# 3. Use the trained model for fast transfer
nst transfer content.jpg style.jpg output.jpg --method fast --model ./checkpoints/final_model.pth
```

## 3. Process a Video

```bash
# Standard processing (best quality)
nst video input.mp4 output.mp4 --model ./checkpoints/final_model.pth

# Real-time mode (faster)
nst video input.mp4 output.mp4 --model ./checkpoints/final_model.pth --realtime
```

## 4. Webcam Style Transfer

```bash
# Live style transfer from webcam
nst webcam --model ./checkpoints/final_model.pth --fps 30

# Press 'q' to quit
```

## 5. Batch Process Images

```bash
# Process entire directory
nst batch ./input_images ./output_images --model ./checkpoints/final_model.pth --recursive
```

## Common Commands

### Check System Info
```bash
nst info
```

### Library Management
```bash
# Add style to library
nst library add "My Style" ./model.pth --style-image style.jpg --tags "artistic,custom"

# List all styles
nst library list

# Show library stats
nst library stats
```

### Get Help
```bash
# General help
nst --help

# Command-specific help
nst transfer --help
nst video --help
nst train --help
```

## Using Python API

```python
from nst_suite import FastStyleTransfer
from nst_suite.utils import load_image, save_image

# Load model
transfer = FastStyleTransfer(model_path='./checkpoints/final_model.pth')

# Load image
content = load_image('content.jpg')

# Apply style transfer
stylized = transfer.transfer(content)

# Save result
save_image(stylized, 'output.jpg')
```

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Follow Windows 11 setup guide**: [docs/WINDOWS_11_SETUP.md](docs/WINDOWS_11_SETUP.md)
3. **Try the examples**: `python examples/basic_image_transfer.py`
4. **Explore the API**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

## Troubleshooting

**CUDA not available?**
```bash
# Install CUDA toolkit and cuDNN
# See docs/WINDOWS_11_SETUP.md for detailed instructions
```

**Out of memory?**
```bash
# Reduce batch size
nst video input.mp4 output.mp4 --model model.pth --batch-size 2
```

**Slow processing?**
```bash
# Use resize factor
nst webcam --model model.pth --resize-factor 0.5
```

## Getting Help

- **Documentation**: Check [README.md](README.md) and [docs/](docs/)
- **Examples**: See [examples/](examples/)
- **Issues**: https://github.com/coco11211/ai-medical-diagnosis/issues

Happy styling! 🎨
