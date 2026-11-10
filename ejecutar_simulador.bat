@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ============================================================================
echo   SIMULADOR DE PIROLISIS DE QUEBRACHO Y CONDENSACION PASIVA (TFI)
echo ============================================================================
echo Directorio de trabajo: "%CD%"
echo.

set "PYTHON_EXE="

:: 1. Buscar en instalacion local del usuario actual (%LOCALAPPDATA%\Python)
for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-*") do (
    if exist "%%D\python.exe" set "PYTHON_EXE=%%D\python.exe"
)

:: 2. Buscar en %LOCALAPPDATA%\Programs\Python
if not defined PYTHON_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" set "PYTHON_EXE=%%D\python.exe"
    )
)

:: 3. Fallback a python del PATH general
if not defined PYTHON_EXE (
    set "PYTHON_EXE=python"
)

echo Usando interprete: "%PYTHON_EXE%"
echo Verificando librerias necesarias (streamlit, pandas, numpy, plotly)...
"%PYTHON_EXE%" -c "import streamlit, pandas, numpy, plotly" 2>nul
if errorlevel 1 (
    echo Instalando librerias necesarias desde requirements.txt...
    "%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
)

echo.
echo Iniciando servidor interactivo Streamlit...
"%PYTHON_EXE%" -m streamlit run "%~dp0app.py"
pause
