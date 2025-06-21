@echo off
:: ai_venv.cmd - Script to create a Python virtual environment with GPU-enabled PyTorch and TensorFlow

setlocal enabledelayedexpansion

:: Configuration
set "ENV_NAME=ai_venv"
set "PYTHON_VERSION=3.12.7"
set "ENV_PATH=%~dp0%ENV_NAME%"

:: Check if the virtual environment already exists
if exist "%ENV_PATH%\Scripts\activate.bat" (
    echo Environment '%ENV_NAME%' already exists at %ENV_PATH%
    
    :: Activate the existing environment
    echo Activating existing environment...
    call "%ENV_PATH%\Scripts\activate.bat"
) else (
    :: Create a new virtual environment
    echo Creating new Python %PYTHON_VERSION% virtual environment: %ENV_NAME%
    
    :: Check Python installation
    set "PYTHON_INSTALLED=0"
    set "PYTHON_CMD="
    
    :: Try python command
    python --version 2>nul | findstr "%PYTHON_VERSION%" >nul
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_INSTALLED=1"
        set "PYTHON_CMD=python"
    )
    
    :: Try py launcher with specific version
    if !PYTHON_INSTALLED! EQU 0 (
        py -%PYTHON_VERSION:~0,1%.%PYTHON_VERSION:~2,1% --version 2>nul | findstr "%PYTHON_VERSION%" >nul
        if !ERRORLEVEL! EQU 0 (
            set "PYTHON_INSTALLED=1"
            set "PYTHON_CMD=py -%PYTHON_VERSION:~0,1%.%PYTHON_VERSION:~2,1%"
        )
    )
    
    if !PYTHON_INSTALLED! EQU 0 (
        echo Error: Python %PYTHON_VERSION% is not installed. Please install it first.
        exit /b 1
    )
    
    :: Create the virtual environment
    echo Creating virtual environment with !PYTHON_CMD!...
    if "!PYTHON_CMD!"=="python" (
        python -m venv "%ENV_PATH%"
    ) else (
        py -%PYTHON_VERSION:~0,1%.%PYTHON_VERSION:~2,1% -m venv "%ENV_PATH%"
    )
    
    :: Activate the environment
    echo Activating environment...
    call "%ENV_PATH%\Scripts\activate.bat"
    
    :: Upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip
    
    :: Install PyTorch with CUDA support
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    :: Install TensorFlow with GPU support
    echo Installing TensorFlow with GPU support...
    pip install tensorflow
    
    :: Install other common ML packages
    echo Installing additional ML packages...
    pip install numpy pandas matplotlib scikit-learn
    
    echo Environment setup complete!
)

:: Create GPU test script
set "TEMP_SCRIPT=%TEMP%\test_gpu.py"
echo import os > "%TEMP_SCRIPT%"
echo os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow logging >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo print("="*50) >> "%TEMP_SCRIPT%"
echo print("GPU AVAILABILITY TEST") >> "%TEMP_SCRIPT%"
echo print("="*50) >> "%TEMP_SCRIPT%"
echo print("\nPython Information:") >> "%TEMP_SCRIPT%"
echo import sys >> "%TEMP_SCRIPT%"
echo print(f"Python version: {sys.version}") >> "%TEMP_SCRIPT%"
echo print(f"Python executable: {sys.executable}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo # Test PyTorch GPU >> "%TEMP_SCRIPT%"
echo print("\n"+"="*30) >> "%TEMP_SCRIPT%"
echo print("PyTorch GPU Test:") >> "%TEMP_SCRIPT%"
echo print("="*30) >> "%TEMP_SCRIPT%"
echo try: >> "%TEMP_SCRIPT%"
echo     import torch >> "%TEMP_SCRIPT%"
echo     print(f"PyTorch version: {torch.__version__}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo     # Check PyTorch GPU >> "%TEMP_SCRIPT%"
echo     gpu_available = torch.cuda.is_available() >> "%TEMP_SCRIPT%"
echo     print(f"PyTorch CUDA available: {gpu_available}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo     if gpu_available: >> "%TEMP_SCRIPT%"
echo         device_count = torch.cuda.device_count() >> "%TEMP_SCRIPT%"
echo         print(f"Number of GPU devices: {device_count}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo         for i in range(device_count): >> "%TEMP_SCRIPT%"
echo             device_name = torch.cuda.get_device_name(i) >> "%TEMP_SCRIPT%"
echo             print(f"  GPU {i}: {device_name}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo         # Test tensor creation on GPU >> "%TEMP_SCRIPT%"
echo         print("\nCreating PyTorch tensor on GPU...") >> "%TEMP_SCRIPT%"
echo         x = torch.tensor([1.0, 2.0, 3.0], device='cuda') >> "%TEMP_SCRIPT%"
echo         print(f"Tensor created: {x}") >> "%TEMP_SCRIPT%"
echo         print(f"Tensor device: {x.device}") >> "%TEMP_SCRIPT%"
echo         print("PyTorch GPU test successful!") >> "%TEMP_SCRIPT%"
echo     else: >> "%TEMP_SCRIPT%"
echo         print("PyTorch cannot access GPU.") >> "%TEMP_SCRIPT%"
echo except Exception as e: >> "%TEMP_SCRIPT%"
echo     print(f"Error testing PyTorch: {e}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo # Test TensorFlow GPU >> "%TEMP_SCRIPT%"
echo print("\n"+"="*30) >> "%TEMP_SCRIPT%"
echo print("TensorFlow GPU Test:") >> "%TEMP_SCRIPT%"
echo print("="*30) >> "%TEMP_SCRIPT%"
echo try: >> "%TEMP_SCRIPT%"
echo     import tensorflow as tf >> "%TEMP_SCRIPT%"
echo     print(f"TensorFlow version: {tf.__version__}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo     # Check TensorFlow GPU >> "%TEMP_SCRIPT%"
echo     gpu_devices = tf.config.list_physical_devices('GPU') >> "%TEMP_SCRIPT%"
echo     print(f"TensorFlow GPU devices: {gpu_devices}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo     if gpu_devices: >> "%TEMP_SCRIPT%"
echo         for device in gpu_devices: >> "%TEMP_SCRIPT%"
echo             print(f"  {device}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo         # Test with TensorFlow >> "%TEMP_SCRIPT%"
echo         print("\nCreating TensorFlow tensor on GPU...") >> "%TEMP_SCRIPT%"
echo         with tf.device('/GPU:0'): >> "%TEMP_SCRIPT%"
echo             x = tf.constant([1.0, 2.0, 3.0]) >> "%TEMP_SCRIPT%"
echo             print(f"Tensor created: {x}") >> "%TEMP_SCRIPT%"
echo         print("TensorFlow GPU test successful!") >> "%TEMP_SCRIPT%"
echo     else: >> "%TEMP_SCRIPT%"
echo         print("TensorFlow cannot access GPU.") >> "%TEMP_SCRIPT%"
echo except Exception as e: >> "%TEMP_SCRIPT%"
echo     print(f"Error testing TensorFlow: {e}") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo print("\n"+"="*50) >> "%TEMP_SCRIPT%"
echo print("GPU TEST COMPLETE") >> "%TEMP_SCRIPT%"
echo print("="*50) >> "%TEMP_SCRIPT%"

:: Run the GPU test script
echo Running GPU availability test...
python "%TEMP_SCRIPT%"

:: Clean up
del "%TEMP_SCRIPT%"

echo.
echo Done! The '%ENV_NAME%' environment is active.
echo To deactivate the environment, run: deactivate

endlocal
