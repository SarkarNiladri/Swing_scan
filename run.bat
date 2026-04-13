@echo off
echo Starting SwingScan...

start "SwingScan API" cmd /k "cd /d %~dp0 && uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3

start "SwingScan UI" cmd /k "cd /d %~dp0 && python -m http.server 5500"

echo Both servers started. You can close this window.