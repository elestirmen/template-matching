@echo off
setlocal
cd /d "%~dp0.."

:: Flight replay (clean re-evaluation) in the visual_navigation_cuda environment (TensorFlow 2.10 + CUDA, .h5 models on the GPU).
:: Resumable: re-run this file after any interruption. Start it ONCE; a second copy refuses to start.
set "VN_ENV=%USERPROFILE%\anaconda3\envs\visual_navigation_cuda"
set "VN_PY=%VN_ENV%\python.exe"
if not exist "%VN_PY%" (
    echo visual_navigation_cuda ortami bulunamadi: %VN_PY%
    pause
    exit /b 1
)
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%VN_ENV%"
set "VN_BACKEND=keras"
set "TF_CPP_MIN_LOG_LEVEL=2"
echo [flight_replay] Python: %VN_PY%

:: piexif (pure Python EXIF reader) is the only extra package the replay needs
"%VN_PY%" -c "import piexif" 2>nul
if errorlevel 1 "%VN_PY%" -m pip install piexif

:: 0) TensorFlow/GPU check and agreement with the reference outputs
"%VN_PY%" flight_replay\keras_check.py
if errorlevel 1 (
    echo.
    echo KERAS KONTROLU BASARISIZ. Bu pencerenin ekran goruntusunu gonderin.
    pause
    exit /b 1
)
:: 1) structural maps
"%VN_PY%" flight_replay\run_all.py --only maps
if errorlevel 1 (
    echo.
    echo HARITA URETIMI BASARISIZ. Bu pencerenin ekran goruntusunu gonderin.
    pause
    exit /b 1
)
:: 2) main replay, 3) low-texture probe and sensitivity runs
"%VN_PY%" flight_replay\run_all.py --only main --jobs 4
if errorlevel 1 (
    pause
    exit /b 1
)
"%VN_PY%" flight_replay\run_all.py --only sens --jobs 4
echo.
echo Replay tamamlandi. Sonuclar: flight_replay\runs\*.csv
pause
endlocal
