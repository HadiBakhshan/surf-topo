@echo off
setlocal enabledelayedexpansion

set "STEP=%~1"
set "THREADS=%~2"
if "%THREADS%"=="" set "THREADS=4"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"

set "BUILD_DIR=%PROJECT_ROOT%\build"
set "BIN_DIR=%PROJECT_ROOT%\bin"

where python >nul 2>&1 || (
    echo [ERROR] Python not found in PATH.
    exit /b 1
)

for /f "delims=" %%i in ('where python') do (
    set "PYTHON_EXECUTABLE=%%i"
    goto :PY_FOUND
)

:PY_FOUND
echo [INFO] Using Python interpreter: %PYTHON_EXECUTABLE%

REM --------------------------
REM Clean
REM --------------------------
if "%STEP%"=="" goto BUILD_CLEAN
if /i "%STEP%"=="clean" goto BUILD_CLEAN
if /i "%STEP%"=="build" goto BUILD_START

echo Usage: configure.bat [clean^|build] [threads]
exit /b 1

:BUILD_CLEAN
echo [CLEAN] Removing build/ and bin/ directories...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%BIN_DIR%" rmdir /s /q "%BIN_DIR%"
if /i "%STEP%"=="clean" exit /b 0

REM --------------------------
REM Build
REM --------------------------
:BUILD_START
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

echo [CMAKE] Configuring project...
cmake ^
  -G "Visual Studio 17 2022" ^
  "%PROJECT_ROOT%"

if errorlevel 1 exit /b 1

echo [CMAKE] Building Release with %THREADS% threads...
cmake --build . --config Release --parallel %THREADS%

if errorlevel 1 exit /b 1

echo [DONE] Build complete.
exit /b 0
