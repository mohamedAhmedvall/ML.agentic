@echo off
setlocal
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0start.py" %*
    exit /b
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0start.py" %*
    exit /b
)
echo Python 3.11 ou plus recent est requis. Installez Python puis relancez start.cmd.
exit /b 1
