@echo off
rem Start the OpenVINO embedding server (OpenAI-compatible).
rem Usage: start-embed.cmd ^<model-dir^> [port]
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 ^<model-dir^> [port]
  exit /b 1
)
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8010"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
python "%~dp0embed_server.py" --model-dir "%~1" --port %PORT%
