# AI Virtual Environment Setup for Windows

This directory contains scripts to set up a Python virtual environment with GPU-enabled PyTorch and TensorFlow on Windows systems.

## Requirements

- Windows 10 or later
- Python 3.12.7 installed
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit 12.1 or compatible version installed
- NVIDIA cuDNN installed (for TensorFlow)

## Setup Instructions

### Using PowerShell (Recommended)

1. Open Windows PowerShell
2. Navigate to this directory
3. Run the script:
   ```powershell
   .\ai_venv.ps1
   ```
   
If you encounter execution policy restrictions, you may need to run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\ai_venv.ps1
```

### Using Command Prompt (CMD)

1. Open Command Prompt
2. Navigate to this directory
3. Run the script:
   ```cmd
   ai_venv.cmd
   ```

## What the Scripts Do

- Check if the virtual environment already exists
- If not, create a new virtual environment with Python 3.12.7
- Install PyTorch with CUDA support
- Install TensorFlow with GPU support
- Install other common ML packages (numpy, pandas, matplotlib, scikit-learn)
- Activate the environment
- Run a test script to verify GPU functionality for both PyTorch and TensorFlow

If the environment already exists, the script will only activate it and run the GPU test.

## Deactivating the Environment

To deactivate the virtual environment, simply run:
```
deactivate
```

## Troubleshooting

### Common Issues

1. **Python version not found**
   - Make sure Python 3.12.7 is installed and available in your PATH

2. **GPU not detected**
   - Verify that your NVIDIA drivers are up to date
   - Ensure CUDA Toolkit 12.1 is properly installed
   - Check if your GPU is CUDA-compatible

3. **Installation errors**
   - If PyTorch or TensorFlow installation fails, try manually installing with:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install tensorflow
     ```

4. **Execution policy restrictions in PowerShell**
   - If you get execution policy errors, try:
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```

For more information on GPU setup for these libraries, visit:
- PyTorch: https://pytorch.org/get-started/locally/
- TensorFlow: https://www.tensorflow.org/install/gpu
