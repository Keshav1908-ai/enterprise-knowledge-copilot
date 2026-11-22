@echo off
REM Stops the containers. Data (documents, vectors, chats) stays in the Docker volume.
cd /d "%~dp0"
docker compose down
echo Stopped. Run start-windows.bat to start again.
pause
