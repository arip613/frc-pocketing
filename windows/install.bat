@echo off
setlocal
rem One-time setup of the FRC plate pocketer on Windows. Double-click this file.
cd /d "%~dp0.."
echo Setting up the FRC plate pocketer. The first time takes a few minutes.
echo.
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, which sets up Python for this tool...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo.
  echo uv was installed but this window can't find it yet.
  echo Close this window and double-click install.bat again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  uv venv --python 3.12 .venv
  if errorlevel 1 goto failed
)
uv pip install --python ".venv\Scripts\python.exe" -e .
if errorlevel 1 goto failed
echo.
echo Done. Double-click start.bat in the windows folder to run the pocketer.
pause
exit /b 0

:failed
echo.
echo Setup failed. Copy the messages above and send them to whoever shared this tool with you.
pause
exit /b 1
