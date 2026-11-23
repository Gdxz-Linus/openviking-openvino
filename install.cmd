@echo off & setlocal EnableExtensions & set "PYEXE="
for %%P in ("%OPENVIKING_HOME%\venv\Scripts\python.exe" "%OV_HOME%\venv\Scripts\python.exe" "%~dp0..\..\venv\Scripts\python.exe" "%~dp0..\venv\Scripts\python.exe") do if not defined PYEXE if exist "%%~P" set "PYEXE=%%~P"
if not defined PYEXE for %%P in (python.exe) do set "PYEXE=%%~$PATH:P"
if not defined PYEXE (echo Python not found & pause & exit /b 1)
"%PYEXE%" -I "%~dp0install.py" %* & pause
