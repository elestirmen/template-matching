@echo off
setlocal
cd /d "%~dp0"

:: CUDA ortami (visual_navigation_cuda) ile baslatmayi onlemek icin ortam
:: acikca secilir ve python.exe tam yoldan cagrilir.
set "VN_ENV=%USERPROFILE%\anaconda3\envs\visual_navigation"
set "VN_PY=%VN_ENV%\python.exe"
if not exist "%VN_PY%" (
    echo visual_navigation ortami bulunamadi: %VN_PY%
    pause
    exit /b 1
)

:: Aktivasyon: conda ortam degiskenleri (GDAL/PROJ vb.) icin gerekli.
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%VN_ENV%"
if errorlevel 1 (
    echo visual_navigation ortami etkinlestirilemedi.
    pause
    exit /b 1
)

echo [mission control ui] Python: %VN_PY%
"%VN_PY%" konum_ui_qt.py
if errorlevel 1 pause
endlocal
