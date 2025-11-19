# Examples

This directory contains example scripts demonstrating various features of the Neural Style Transfer Suite.

## Available Examples

### 1. Basic Image Transfer (`basic_image_transfer.py`)

Demonstrates basic style transfer on images using both methods:
- Optimization-based transfer (high quality)
- Fast feed-forward transfer (real-time)

**Usage:**
```bash
python examples/basic_image_transfer.py
```

**Requirements:**
- `content.jpg` - content image
- `style.jpg` - style image
- (Optional) Trained model at `./checkpoints/final_model.pth`

---

### 2. Video Processing (`video_processing.py`)

Shows different methods for processing videos:
- Standard batch processing
- Real-time processing
- 4K video support

**Usage:**
```bash
python examples/video_processing.py
```

**Requirements:**
- `input.mp4` - input video
- Trained model at `./checkpoints/final_model.pth`

---

### 3. Webcam Demo (`webcam_demo.py`)

Real-time style transfer from webcam with FPS display.

**Usage:**
```bash
python examples/webcam_demo.py
```

**Requirements:**
- Webcam connected to your computer
- Trained model at `./checkpoints/final_model.pth`

---

### 4. Batch Processing (`batch_processing.py`)

Efficiently process multiple images and apply multiple styles.

**Usage:**
```bash
python examples/batch_processing.py
```

**Requirements:**
- Directory with images at `./input_images`
- Trained model(s)

---

### 5. Style Interpolation (`style_interpolation.py`)

Demonstrates style blending and morphing:
- Interpolation between styles
- Smooth transitions
- Spatial style blending

**Usage:**
```bash
python examples/style_interpolation.py
```

**Requirements:**
- `content.jpg` - content image
- Multiple trained models (at least 2)

---

### 6. Library Management (`library_management.py`)

Shows how to organize and manage style models using the library system.

**Usage:**
```bash
python examples/library_management.py
```

**Requirements:**
- Trained models to add to library

---

## Quick Start

### Training Your First Model

Before running most examples, you'll need to train a style transfer model:

```bash
# 1. Download a dataset (e.g., COCO or any image collection)
# 2. Prepare a style image
# 3. Train the model

nst train style.jpg ./path/to/dataset --epochs 2 --batch-size 4
```

This will create a model at `./checkpoints/final_model.pth`.

### Running All Examples

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (one-time setup)
nst train your_style.jpg ./dataset

# Run examples
python examples/basic_image_transfer.py
python examples/video_processing.py
python examples/webcam_demo.py
python examples/batch_processing.py
python examples/style_interpolation.py
python examples/library_management.py
```

## Example Outputs

Each example creates outputs in specific directories:

- `./outputs/` - Main output directory
- `./outputs/batch/` - Batch processing results
- `./outputs/interpolation/` - Style interpolation results
- `./outputs/multi_style/` - Multi-style results
- `./style_library/` - Style library data

## Tips

### For Better Results

1. **Image Quality**: Use high-quality images (minimum 512x512)
2. **Training Data**: Use diverse content images for training
3. **Training Time**: More epochs = better quality (try 4-6 epochs)
4. **GPU**: Enable CUDA for much faster processing

### Performance

1. **Reduce batch size** if running out of memory
2. **Use resize factor** for faster webcam/video processing
3. **Enable half-precision** for modern NVIDIA GPUs
4. **Close other applications** when processing large videos

### Troubleshooting

**Issue**: Module not found errors
```bash
# Solution: Install in development mode
pip install -e .
```

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size or image resolution
nst video input.mp4 output.mp4 --model model.pth --batch-size 2
```

**Issue**: Webcam not working
```bash
# Solution: Try different camera ID
python examples/webcam_demo.py --camera 1
```

## Next Steps

After running the examples:

1. Experiment with different style images
2. Train models with your own artistic styles
3. Create style collections in the library
4. Process your own videos and images
5. Explore the API for custom applications

## Additional Resources

- [Main README](../README.md) - Full documentation
- [Windows 11 Setup Guide](../docs/WINDOWS_11_SETUP.md) - Installation guide
- [API Documentation](../docs/) - Detailed API reference

## Contributing

Found an issue or want to add an example? Please submit a pull request!
