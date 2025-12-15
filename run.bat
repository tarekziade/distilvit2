@echo off
REM DistilVit - Image Captioning Model Training
REM Batch file wrapper for Windows
REM This is a simple wrapper that calls the PowerShell script

if "%1"=="" (
    powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1" help
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
)
