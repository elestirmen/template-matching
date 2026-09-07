@echo off
setlocal
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate visual_navigation
if errorlevel 1 (
    echo visual_navigation ortami etkinlestirilemedi.
    pause
    exit /b 1
)
python konum_ui_qt.py
if errorlevel 1 pause
endlocal
