@echo off
title YokeClassifier - Industrial Visual Inspection
cd /d "%~dp0"
echo ========================================================
echo   YokeClassifier - Real-Time Visual Inspection Station
echo ========================================================
echo.
echo Launching inspection pipeline...
python phase7_live\live_inspection.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application stopped with error code %errorlevel%.
    echo Please verify camera IP, network connection, and dependencies.
    pause
)
