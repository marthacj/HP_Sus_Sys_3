@echo off
REM Navigate to the directory where this script is located
cd /d %~dp0

REM Echo the current directory and the command being run
echo Current directory: %CD%
echo Running command: "%~dp0ollama.exe" serve

REM Set environment variable for model path
set OLLAMA_MODELS=%~dp0models
echo Ollama Model path set to: %OLLAMA_MODELS%

REM Start the ollama server
start "" "%~dp0ollama.exe" serve
