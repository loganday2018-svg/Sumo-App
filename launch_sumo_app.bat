@echo off
setlocal
REM Launches the Streamlit-based Sumo companion app.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo Activating local virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo No .venv found, using system Python.
)

echo Starting Streamlit app...
python -m streamlit run sumo_companion_app.py

echo.
echo Streamlit closed. Press any key to exit.
pause >nul
