@echo off
setlocal
cd /d "%~dp0"

start "DeepRead Backend" cmd /k ""%~dp0run-backend.cmd""
start "DeepRead Frontend" cmd /k ""%~dp0run-frontend.cmd""

echo Started DeepRead services in separate windows.
echo Backend:  http://localhost:8000/health
echo Frontend: http://localhost:3000

