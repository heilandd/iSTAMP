@echo off
REM Build a Windows .exe with PyInstaller.
REM
REM Usage (from the 10_cross_platform_app folder):
REM   packaging\build_windows.bat
REM
REM The model checkpoint (best_model_full.pt) is NOT auto-bundled. Either copy
REM it next to the .exe and set RMC_CHECKPOINT, or add an --add-data line below.

setlocal
cd /d "%~dp0\.."

set APP_NAME=RamanMolecularClassifier
set ENTRY=src\ramanmolecularclassifier\app.py

python -m pip install --upgrade pyinstaller

pyinstaller ^
  --noconfirm --clean --windowed ^
  --name "%APP_NAME%" ^
  --paths src ^
  --collect-all torch_geometric ^
  --collect-submodules ramanmolecularclassifier ^
  --hidden-import scipy.interpolate ^
  --hidden-import openpyxl ^
  --hidden-import pydicom ^
  --hidden-import tifffile ^
  %ENTRY%

echo Built dist\%APP_NAME%\%APP_NAME%.exe
echo Reminder: provide best_model_full.pt via RMC_CHECKPOINT or place beside the exe.
endlocal
