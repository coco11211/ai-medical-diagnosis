# Windows 11 Installation and Configuration Guide

## Overview

This guide provides detailed instructions for installing and configuring the AI Medical Diagnosis System on Windows 11.

## Prerequisites

### System Requirements

- **Operating System**: Windows 11 (64-bit)
- **Processor**: Intel Core i5 or AMD Ryzen 5 (or better)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 10GB free space (SSD recommended)
- **GPU** (optional): NVIDIA GPU with 6GB+ VRAM
- **Internet**: Required for downloading dependencies

### Required Software

1. **Python 3.9 or higher**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Git** (optional, for cloning repository)
   - Download from: https://git-scm.com/download/win

3. **CUDA Toolkit** (optional, for GPU support)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Required version: 11.8 or higher

## Installation Methods

### Method 1: Automated Setup (Recommended)

1. Open Command Prompt or PowerShell as Administrator
2. Navigate to the project directory:
```cmd
cd path\to\ai-medical-diagnosis
```

3. Run the setup script:
```cmd
scripts\setup_windows.bat
```

4. Wait for installation to complete (5-15 minutes depending on internet speed)

5. Edit configuration files:
   - `.env` - Environment variables
   - `config\config.yaml` - System configuration

### Method 2: Manual Setup

1. **Create Virtual Environment**
```cmd
python -m venv venv
```

2. **Activate Virtual Environment**
```cmd
venv\Scripts\activate
```

3. **Upgrade pip**
```cmd
python -m pip install --upgrade pip
```

4. **Install Dependencies**
```cmd
pip install -r requirements.txt
```

5. **Download NLTK Data**
```cmd
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

6. **Create Directories**
```cmd
mkdir data\raw data\processed data\models logs output
```

7. **Copy Configuration**
```cmd
copy .env.example .env
```

## GPU Setup (NVIDIA)

### Check GPU Availability

1. Open Command Prompt
2. Run:
```cmd
nvidia-smi
```

3. Verify CUDA version matches PyTorch requirements

### Configure for GPU

1. Edit `.env`:
```env
DEVICE=cuda
```

2. Edit `config\config.yaml`:
```yaml
device: "cuda"
```

3. Test GPU in Python:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Running the System

### Start API Server

#### Option 1: Using Batch Script
```cmd
scripts\start_server.bat
```

#### Option 2: Manual Start
```cmd
venv\Scripts\activate
python -m src.api.main
```

The API will be available at:
- Main API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Run Tests

```cmd
scripts\run_tests.bat
```

Or manually:
```cmd
venv\Scripts\activate
pytest
```

## Windows Service Installation

To run the system as a Windows service:

1. Open PowerShell as Administrator
2. Navigate to project directory
3. Run:
```powershell
.\scripts\install_service.ps1
```

4. Start the service:
```powershell
Start-Service -Name AIMedicalDiagnosis
```

5. Check service status:
```powershell
Get-Service -Name AIMedicalDiagnosis
```

## Windows Firewall Configuration

### Allow API Server

1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Click "Inbound Rules" → "New Rule"
4. Select "Port"
5. Enter port 8000 (or your configured port)
6. Allow the connection
7. Name the rule "AI Medical Diagnosis API"

### Command Line Method

```powershell
New-NetFirewallRule -DisplayName "AI Medical Diagnosis API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

## Windows Defender Exclusions

To improve performance, add exclusions:

1. Open Windows Security
2. Go to "Virus & threat protection"
3. Click "Manage settings"
4. Scroll to "Exclusions"
5. Add folder: `path\to\ai-medical-diagnosis`

### Command Line Method

```powershell
Add-MpPreference -ExclusionPath "C:\path\to\ai-medical-diagnosis"
```

## Task Scheduler Setup

Schedule automated tasks:

1. Open Task Scheduler
2. Create Basic Task
3. Name: "AI Medical Diagnosis"
4. Trigger: When you want it to run
5. Action: Start a program
6. Program: `C:\path\to\ai-medical-diagnosis\scripts\start_server.bat`

## Performance Optimization

### CPU Optimization

1. Edit `config\config.yaml`:
```yaml
api:
  workers: 8  # Set to number of CPU cores
```

2. Set process priority:
```cmd
wmic process where name="python.exe" CALL setpriority "high priority"
```

### Memory Optimization

1. Reduce batch size in config:
```yaml
model:
  batch_size: 16  # Reduce if out of memory
```

2. Enable virtual memory:
   - Settings → System → About → Advanced system settings
   - Performance → Settings → Advanced → Virtual memory

### GPU Optimization

1. Monitor GPU usage:
```cmd
nvidia-smi -l 1
```

2. Adjust model size:
```yaml
model:
  resnet_variant: resnet18  # Use smaller model
  efficientnet_variant: efficientnet-b0
```

## Troubleshooting

### Issue: Python not found

**Solution:**
1. Reinstall Python with "Add to PATH" checked
2. Or add manually to PATH:
   - Settings → System → About → Advanced system settings
   - Environment Variables → Path → Edit
   - Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python39`

### Issue: Permission denied

**Solution:**
1. Run Command Prompt as Administrator
2. Or change folder permissions:
   ```cmd
   icacls "path\to\ai-medical-diagnosis" /grant Users:F /t
   ```

### Issue: Port 8000 already in use

**Solution:**
1. Find process using port:
   ```cmd
   netstat -ano | findstr :8000
   ```

2. Kill process:
   ```cmd
   taskkill /PID <process_id> /F
   ```

3. Or change port in config

### Issue: CUDA out of memory

**Solution:**
1. Reduce batch size in config
2. Use CPU instead: set `device: "cpu"`
3. Close other GPU applications

### Issue: DLL load failed

**Solution:**
1. Install Visual C++ Redistributables:
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Reinstall CUDA toolkit
3. Update GPU drivers

## Security Considerations

### Windows Defender

1. Keep Windows Defender enabled
2. Add exclusions only for project folder
3. Regular security scans

### Network Security

1. Use firewall rules to limit access
2. Consider VPN for remote access
3. Enable HTTPS in production

### Data Security

1. Encrypt patient database
2. Use Windows BitLocker for disk encryption
3. Regular backups to secure location

## Backup and Recovery

### Backup Database

```cmd
copy data\medical_diagnosis.db backups\medical_diagnosis_%date%.db
```

### Automated Backup Script

Create `backup.bat`:
```batch
@echo off
set backup_dir=backups\%date:~-4,4%%date:~-10,2%%date:~-7,2%
mkdir %backup_dir%
copy data\*.db %backup_dir%
echo Backup complete!
```

Schedule with Task Scheduler

## Uninstallation

1. Stop service (if installed):
```powershell
Stop-Service -Name AIMedicalDiagnosis
.\scripts\uninstall_service.ps1
```

2. Remove virtual environment:
```cmd
rmdir /s /q venv
```

3. Remove data (optional):
```cmd
rmdir /s /q data logs output
```

## Support

For Windows 11 specific issues:
- Check Event Viewer for error logs
- Review `logs\` directory
- Check Windows System logs
- Contact support with system information:
  ```cmd
  systeminfo > system_info.txt
  ```

## Additional Resources

- Windows 11 Documentation: https://docs.microsoft.com/windows/
- Python on Windows: https://docs.python.org/3/using/windows.html
- CUDA on Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
