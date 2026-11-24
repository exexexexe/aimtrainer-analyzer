@echo off
rem Runs aimlab_analyzer with the project's venv, from any folder.
setlocal
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m aimlab_analyzer %*
