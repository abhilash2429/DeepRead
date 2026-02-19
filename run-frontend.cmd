@echo off
setlocal
cd /d "%~dp0frontend"

if not exist "node_modules" (
  echo [INFO] node_modules missing. Installing frontend dependencies...
  call npm install
  if errorlevel 1 exit /b 1
)

set "NEXT_PUBLIC_API_BASE=http://localhost:8000"
call npm run dev

