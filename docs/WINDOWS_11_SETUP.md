# Windows 11 Setup Guide

Complete guide for setting up the Neural Style Transfer Suite on Windows 11.

## Prerequisites

### 1. Windows 11 System Requirements

- Windows 11 64-bit (Home, Pro, or Enterprise)
- Administrator access
- Internet connection
- At least 20 GB free disk space

### 2. Install Python

#### Option A: Microsoft Store (Recommended)

1. Open Microsoft Store
2. Search for "Python 3.11"
3. Click "Get" to install
4. Verify installation:
   ```cmd
   python --version
   ```

#### Option B: Python.org

1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Run installer
3. **Important**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify:
   ```cmd
   python --version
   pip --version
   ```

### 3. Install Visual Studio Build Tools (Required)

Some Python packages require C++ compilation:

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Run installer
3. Select "Desktop development with C++"
4. Install (requires ~6 GB)

### 4. Install Git

1. Download [Git for Windows](https://git-scm.com/download/win)
2. Run installer with default settings
3. Verify:
   ```cmd
   git --version
   ```

## GPU Setup (For NVIDIA Graphics Cards)

### Step 1: Check GPU Compatibility

```cmd
# Open PowerShell as Administrator
nvidia-smi
```

If this command works, you have an NVIDIA GPU. Note the CUDA version shown.

### Step 2: Install CUDA Toolkit

1. **Download CUDA Toolkit**
   - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select:
     - Operating System: Windows
     - Architecture: x86_64
     - Version: 11
     - Installer Type: exe (local)

2. **Run CUDA Installer**
   ```cmd
   # Run as Administrator
   cuda_11.8.0_windows.exe
   ```

3. **Choose Express Installation**
   - Installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
   - Wait for installation (10-15 minutes)

4. **Verify Installation**
   ```cmd
   nvcc --version
   ```

### Step 3: Install cuDNN

1. **Download cuDNN**
   - Visit [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Sign in/create NVIDIA Developer account (free)
   - Download cuDNN for CUDA 11.x (ZIP file)

2. **Extract cuDNN**
   - Extract downloaded ZIP file
   - You'll see folders: `bin`, `include`, `lib`

3. **Copy cuDNN Files to CUDA Directory**

   Open PowerShell as Administrator:

   ```powershell
   # Replace paths with your actual paths
   $cudnn = "C:\Users\YourName\Downloads\cudnn-windows-x86_64-8.x.x.x_cuda11"
   $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

   # Copy files
   Copy-Item "$cudnn\bin\*" "$cuda\bin\" -Force
   Copy-Item "$cudnn\include\*" "$cuda\include\" -Force
   Copy-Item "$cudnn\lib\*" "$cuda\lib\x64\" -Force
   ```

### Step 4: Set Environment Variables

1. Open "Environment Variables":
   - Press `Win + X`
   - Select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"

2. Add to PATH (if not already there):
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp`

3. Create new variable (if doesn't exist):
   - Variable name: `CUDA_PATH`
   - Variable value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

## Install Neural Style Transfer Suite

### Step 1: Clone Repository

Open Command Prompt or PowerShell:

```cmd
# Navigate to desired location
cd C:\Users\YourName\Documents

# Clone repository
git clone https://github.com/coco11211/ai-medical-diagnosis.git
cd ai-medical-diagnosis
```

### Step 2: Create Virtual Environment (Recommended)

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Your prompt should now show (venv)
```

### Step 3: Install Dependencies

```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Step 4: Verify Installation

```cmd
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Should print: CUDA available: True

# Check NST Suite
nst info
```

You should see output showing CUDA information and GPU details.

## Common Issues and Solutions

### Issue 1: CUDA Not Available

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:

1. Check PyTorch installation:
   ```cmd
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify CUDA installation:
   ```cmd
   nvcc --version
   nvidia-smi
   ```

3. Check PATH variables

### Issue 2: DLL Load Failed

**Symptoms**: Error about missing DLL files

**Solutions**:

1. Install Visual C++ Redistributables:
   - Download from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   - Install both x86 and x64 versions

2. Reinstall CUDA and cuDNN

### Issue 3: Out of Memory Errors

**Symptoms**: CUDA out of memory during processing

**Solutions**:

1. Reduce batch size:
   ```cmd
   nst video input.mp4 output.mp4 --model model.pth --batch-size 2
   ```

2. Use resize factor:
   ```cmd
   nst webcam --model model.pth --resize-factor 0.5
   ```

3. Close other GPU applications

### Issue 4: Slow Performance

**Solutions**:

1. Ensure GPU is being used:
   ```cmd
   nst info
   ```

2. Use half-precision:
   - Automatically enabled for video processing

3. Update NVIDIA drivers:
   - Use GeForce Experience or download from NVIDIA website

### Issue 5: Module Not Found Errors

**Solutions**:

1. Ensure virtual environment is activated:
   ```cmd
   venv\Scripts\activate
   ```

2. Reinstall dependencies:
   ```cmd
   pip install -r requirements.txt --force-reinstall
   ```

3. Check Python path:
   ```cmd
   where python
   ```

## Performance Optimization for Windows 11

### 1. Enable High Performance Mode

```powershell
# Run as Administrator
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### 2. Disable Windows Visual Effects

1. Press `Win + X`, select "System"
2. Click "Advanced system settings"
3. Under "Performance", click "Settings"
4. Select "Adjust for best performance"
5. Click "Apply"

### 3. Close Background Applications

```cmd
# Open Task Manager (Ctrl+Shift+Esc)
# End unnecessary processes, especially:
# - Web browsers
# - Other GPU applications
# - Background apps
```

### 4. Increase Virtual Memory

1. Press `Win + X`, select "System"
2. Click "Advanced system settings"
3. Under "Performance", click "Settings"
4. Go to "Advanced" tab
5. Under "Virtual memory", click "Change"
6. Uncheck "Automatically manage"
7. Set Initial size: 16384 MB
8. Set Maximum size: 32768 MB
9. Click "Set" and "OK"

## Testing Your Installation

### Quick Test Script

Create a file `test_installation.py`:

```python
import torch
from nst_suite.utils.cuda import print_cuda_info

# Print system information
print("="*50)
print("Testing Neural Style Transfer Suite Installation")
print("="*50)

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Check CUDA utilities
print("\n" + "="*50)
print_cuda_info()

print("\n" + "="*50)
print("Installation test complete!")
print("="*50)
```

Run the test:

```cmd
python test_installation.py
```

### Run Example

```cmd
# Download a test image
curl -o content.jpg https://images.unsplash.com/photo-1506905925346-21bda4d32df4

# Train a simple model (this will take time)
nst train content.jpg ./test_dataset --epochs 1

# Or download a pre-trained model and test
# nst transfer content.jpg style.jpg output.jpg --method fast --model model.pth
```

## Updating the Suite

```cmd
# Activate virtual environment
venv\Scripts\activate

# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e .
```

## Uninstallation

```cmd
# Deactivate virtual environment
deactivate

# Remove virtual environment
rmdir /s /q venv

# Remove repository
cd ..
rmdir /s /q ai-medical-diagnosis
```

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Windows Installation](https://pytorch.org/get-started/locally/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Search existing [GitHub Issues](https://github.com/coco11211/ai-medical-diagnosis/issues)
3. Create a new issue with:
   - Windows version
   - Python version
   - CUDA version
   - Error message
   - Steps to reproduce

## Next Steps

Once installation is complete:

1. Read the [README.md](../README.md) for usage examples
2. Try the examples in `examples/` directory
3. Explore the CLI commands: `nst --help`
4. Train your first custom style model
