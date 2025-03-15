@echo off
echo ResNetV2 Anti-Overfitting Training Script
echo ----------------------------------------
echo 1. Train ResNet18v2
echo 2. Train ResNet50v2 (batch size 64)
echo 3. Train ResNet101v2
echo 4. Exit
echo.

set /p choice=Enter your choice (1-4): 

if "%choice%"=="1" (
    echo Starting ResNet18v2 training...
    call train_antioverfitting.bat
) else if "%choice%"=="2" (
    echo Starting ResNet50v2 training with batch size 64...
    call train_antioverfitting_resnet50.bat
) else if "%choice%"=="3" (
    echo Starting ResNet101v2 training...
    call train_antioverfitting_resnet101.bat
) else if "%choice%"=="4" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter a number between 1 and 4.
    echo.
    goto :choice
)

echo All training completed! 