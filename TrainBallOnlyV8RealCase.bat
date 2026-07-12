@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\GHOST-V3\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set "IMAGE_SOURCE=D:\Dev\cvatpacks\cvat_frames_20260709_173500\images"

set "CVAT_ZIP=%~1"
if "%CVAT_ZIP%"=="" (
  echo CVAT YOLO ZIP dosyasini bu BAT dosyasinin ustune surukleyip birakabilirsin.
  set /p CVAT_ZIP=CVAT ZIP yolu: 
)

if not exist "%CVAT_ZIP%" (
  echo HATA: CVAT ZIP bulunamadi: %CVAT_ZIP%
  pause
  exit /b 1
)

echo.
echo [1/5] Gercek-case CVAT test seti hazirlaniyor...
"%PYTHON_EXE%" -u "saha_ai_project\scripts\17_import_cvat_ball_test_set.py" ^
  --zip "%CVAT_ZIP%" ^
  --images-source "%IMAGE_SOURCE%" ^
  --output "datasets\BallOnly\real_case_test_v1" ^
  --overwrite
if errorlevel 1 goto fail

echo.
echo [2/5] Augmented v3 train dataset kontrol ediliyor...
if not exist "saha_ai_project\datasets\BallOnly\master_v3_augmented\data.yaml" (
  echo master_v3_augmented yok, simdi olusturulacak...
  "%PYTHON_EXE%" -u "saha_ai_project\scripts\16_build_ball_v3_augmented_dataset.py" ^
    --dataset "datasets\BallOnly\master_v1" ^
    --output "datasets\BallOnly\master_v3_augmented" ^
    --seed 20260710 ^
    --negative-augment-ratio 0.25 ^
    --combo-ratio 0.50 ^
    --copy-mode hardlink ^
    --overwrite
  if errorlevel 1 goto fail
) else (
  echo master_v3_augmented mevcut, tekrar uretilmedi.
)

echo.
echo [3/5] Ball v8 real-case dataset kuruluyor...
"%PYTHON_EXE%" -u "saha_ai_project\scripts\18_build_ball_dataset_with_real_test.py" ^
  --base-dataset "datasets\BallOnly\master_v3_augmented" ^
  --real-test "datasets\BallOnly\real_case_test_v1" ^
  --output "datasets\BallOnly\master_v8_realcase" ^
  --copy-mode hardlink ^
  --overwrite
if errorlevel 1 goto fail

echo.
echo [4/5] ball_only_v8 egitimi baslatiliyor...
"%PYTHON_EXE%" -u "saha_ai_project\scripts\15_train_ball_model.py" ^
  --dataset "datasets\BallOnly\master_v8_realcase" ^
  --base-model "models\base\yolo26m.pt" ^
  --name "ball_only_v8" ^
  --epochs 80 ^
  --imgsz 1024 ^
  --batch 2 ^
  --device 0 ^
  --patience 20 ^
  --workers 4
if errorlevel 1 goto fail

echo.
echo [5/5] Gercek-case test splitinde validation/F1 raporu aliniyor...
"%PYTHON_EXE%" -u "saha_ai_project\scripts\09_validate_model.py" ^
  --model "models\trained\ball_only_v8\weights\best.pt" ^
  --dataset "datasets\BallOnly\master_v8_realcase" ^
  --imgsz 1024 ^
  --device 0 ^
  --name "ball_only_v8_realcase_test" ^
  --split test
if errorlevel 1 goto fail

echo.
echo TAMAM: ball_only_v8 egitildi ve gercek-case test raporu alindi.
echo Model: saha_ai_project\models\trained\ball_only_v8\weights\best.pt
echo Rapor: saha_ai_project\experiments\outputs\ball_only_v8_realcase_test
pause
exit /b 0

:fail
echo.
echo HATA: Islem durdu. Ustteki hata mesajina bak.
pause
exit /b 1
