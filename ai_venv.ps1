#!/usr/bin/env pwsh
# ai_venv.ps1 - Script to create a Python virtual environment with GPU-enabled PyTorch and TensorFlow

# Configuration
$envName = "ai_venv"
$pythonVersion = "3.12.7"
$envPath = Join-Path $PSScriptRoot $envName

# Function to check if GPU is available with PyTorch and TensorFlow
$testGpuScript = @"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow logging

print("="*50)
print("GPU AVAILABILITY TEST")
print("="*50)
print("\nPython Information:")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test PyTorch GPU
print("\n"+"="*30)
print("PyTorch GPU Test:")
print("="*30)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check PyTorch GPU
    gpu_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {gpu_available}")
    
    if gpu_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPU devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {device_name}")
            
        # Test tensor creation on GPU
        print("\nCreating PyTorch tensor on GPU...")
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        print(f"Tensor created: {x}")
        print(f"Tensor device: {x.device}")
        print("PyTorch GPU test successful!")
    else:
        print("PyTorch cannot access GPU.")
except Exception as e:
    print(f"Error testing PyTorch: {e}")

# Test TensorFlow GPU
print("\n"+"="*30)
print("TensorFlow GPU Test:")
print("="*30)
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check TensorFlow GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPU devices: {gpu_devices}")
    
    if gpu_devices:
        for device in gpu_devices:
            print(f"  {device}")
        
        # Test with TensorFlow
        print("\nCreating TensorFlow tensor on GPU...")
        with tf.device('/GPU:0'):
            x = tf.constant([1.0, 2.0, 3.0])
            print(f"Tensor created: {x}")
        print("TensorFlow GPU test successful!")
    else:
        print("TensorFlow cannot access GPU.")
except Exception as e:
    print(f"Error testing TensorFlow: {e}")

print("\n"+"="*50)
print("GPU TEST COMPLETE")
print("="*50)
"@

# Check if the virtual environment already exists
if (Test-Path $envPath) {
    Write-Host "Environment '$envName' already exists at $envPath" -ForegroundColor Yellow
    
    # Activate the existing environment
    Write-Host "Activating existing environment..." -ForegroundColor Cyan
    & "$envPath\Scripts\Activate.ps1"
} else {
    # Create a new virtual environment
    Write-Host "Creating new Python $pythonVersion virtual environment: $envName" -ForegroundColor Green
    
    # Check if Python is installed
    try {
        $pythonInstalled = $false
        $pythonCmd = ""
        
        # Try python3 command
        try {
            $ver = python --version
            if ($ver -match $pythonVersion) {
                $pythonInstalled = $true
                $pythonCmd = "python"
            }
        } catch {
            # Python command not found or version doesn't match
        }
        
        # Try py launcher with specific version
        if (-not $pythonInstalled) {
            try {
                $ver = py -$pythonVersion --version
                if ($ver -match $pythonVersion) {
                    $pythonInstalled = $true
                    $pythonCmd = "py -$pythonVersion"
                }
            } catch {
                # Python version not found
            }
        }
        
        if (-not $pythonInstalled) {
            throw "Python $pythonVersion is not installed. Please install it first."
        }
        
        # Create the virtual environment
        Write-Host "Creating virtual environment with $pythonCmd..." -ForegroundColor Cyan
        if ($pythonCmd -eq "python") {
            & python -m venv $envPath
        } else {
            & py -$pythonVersion -m venv $envPath
        }
        
        # Activate the environment
        Write-Host "Activating environment..." -ForegroundColor Cyan
        & "$envPath\Scripts\Activate.ps1"
        
        # Upgrade pip
        Write-Host "Upgrading pip..." -ForegroundColor Cyan
        pip install --upgrade pip
        
        # Install PyTorch with CUDA support
        Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        # Install TensorFlow with GPU support
        Write-Host "Installing TensorFlow with GPU support..." -ForegroundColor Cyan
        pip install tensorflow
        
        # Install other common ML packages
        Write-Host "Installing additional ML packages..." -ForegroundColor Cyan
        pip install numpy pandas matplotlib scikit-learn
        
        Write-Host "Environment setup complete!" -ForegroundColor Green
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        exit 1
    }
}

# Save the GPU test script to a temporary file
$tempScriptPath = Join-Path $env:TEMP "test_gpu.py"
$testGpuScript | Out-File -FilePath $tempScriptPath -Encoding utf8

# Run the GPU test script
Write-Host "Running GPU availability test..." -ForegroundColor Cyan
python $tempScriptPath

# Clean up
Remove-Item $tempScriptPath -Force

Write-Host "`nDone! The '$envName' environment is active." -ForegroundColor Green
Write-Host "To deactivate the environment, run: deactivate" -ForegroundColor Yellow
