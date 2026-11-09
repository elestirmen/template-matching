@echo off
setlocal
:: Switch onnxruntime in the visual_navigation env to the DirectML build (NVIDIA GPU through DirectX 12).
set "VN_PY=%USERPROFILE%\anaconda3\envs\visual_navigation\python.exe"
if not exist "%VN_PY%" (
    echo visual_navigation ortami bulunamadi: %VN_PY%
    pause
    exit /b 1
)
"%VN_PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
"%VN_PY%" -m pip install onnxruntime-directml
"%VN_PY%" "%~dp0gpu_test_ort.py"
echo.
echo "provider: DmlExecutionProvider" yaziyorsa GPU hazir. Simdi run_replay.bat dosyasini baslatin.
pause
endlocal
