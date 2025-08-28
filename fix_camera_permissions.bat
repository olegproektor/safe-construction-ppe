@echo off
echo Fixing Camera Permissions for Windows...
echo.
echo This script will enable camera access at the system level.
echo Please run as Administrator for best results.
echo.
pause

REM Enable camera access for all users
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam" /v Value /t REG_SZ /d Allow /f

REM Enable camera for desktop apps
reg add "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam" /v Value /t REG_SZ /d Allow /f

REM Enable camera globally
reg add "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged" /v Value /t REG_SZ /d Allow /f

echo.
echo Camera permissions have been updated in the registry.
echo Please restart your browser and try again.
echo.
pause