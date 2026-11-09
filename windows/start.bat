@echo off
setlocal
rem Starts the FRC plate pocketer and opens it in your browser. Double-click this file.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo The pocketer isn't set up yet. Double-click install.bat in the windows folder first.
  pause
  exit /b 1
)
set FRCPOCKET_OPEN_BROWSER=1
echo Starting the FRC plate pocketer. This takes about a minute.
echo Your browser opens at http://127.0.0.1:7861 when it is ready.
echo Keep this window open while you use it. Close it to stop the pocketer.
echo.
".venv\Scripts\python.exe" -X utf8 -I -m frcpocket.web
echo.
echo The pocketer has stopped.
pause
