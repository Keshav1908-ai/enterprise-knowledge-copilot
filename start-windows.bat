@echo off
REM Enterprise Knowledge Copilot - one-click start for Windows (requires Docker Desktop)
setlocal
cd /d "%~dp0"
title Knowledge Copilot

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created .env from .env.example. Edit it later to add an LLM API key.
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running yet. Start Docker Desktop, wait until it says "Engine running", then run this file again.
  pause
  exit /b 1
)

echo.
echo Building and starting containers. The first build downloads Python, PyTorch and the
echo embedding model, so it can take 5-15 minutes. Later starts take seconds.
echo.
docker compose up --build -d
if errorlevel 1 (
  echo.
  echo Docker Compose failed. Scroll up for the error, or run: docker compose logs backend
  pause
  exit /b 1
)

echo.
echo Waiting for the backend to become healthy...
for /l %%i in (1,1,60) do (
  curl -fs http://localhost:8000/health >nul 2>&1 && goto ready
  timeout /t 3 /nobreak >nul
)
echo The backend did not report healthy within 3 minutes. Check: docker compose logs backend
pause
exit /b 1

:ready
echo.
echo Knowledge Copilot is running:
echo   App       http://localhost:8080
echo   API docs  http://localhost:8000/docs
start "" http://localhost:8080
echo.
echo Stop it with stop-windows.bat (your documents and chats are kept).
pause
