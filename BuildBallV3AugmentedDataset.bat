@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\GHOST-V3\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Ball-only v3 augmented dataset hazirlaniyor...
echo Kaynak : saha_ai_project\datasets\BallOnly\master_v1
echo Cikti  : saha_ai_project\datasets\BallOnly\master_v3_augmented
echo.

"%PYTHON_EXE%" -u "saha_ai_project\scripts\16_build_ball_v3_augmented_dataset.py" ^
  --dataset "datasets\BallOnly\master_v1" ^
  --output "datasets\BallOnly\master_v3_augmented" ^
  --seed 20260710 ^
  --negative-augment-ratio 0.25 ^
  --combo-ratio 0.50 ^
  --copy-mode hardlink ^
  --overwrite

echo.
if errorlevel 1 (
  echo HATA: Dataset olusturma basarisiz oldu.
) else (
  echo TAMAM: Ball v3 augmented dataset hazir.
  echo data.yaml: saha_ai_project\datasets\BallOnly\master_v3_augmented\data.yaml
)
pause
