@echo off
setlocal

if "%ARCH%"=="" set ARCH=-arch=sm_86
set FLAGS=-O3 -allow-unsupported-compiler -Xcompiler "/O2 /W3 /utf-8 /wd4819 /wd4996 /D_CRT_SECURE_NO_WARNINGS" -Xlinker "/STACK:8388608 /DELAYLOAD:nvcuda.dll" -ldelayimp -Iinclude

if exist src\tkm.rc (
    where rc.exe >nul 2>nul
    if not errorlevel 1 (
        echo [Tokamak] Embedding tkm.ico into Windows resource...
        rc.exe /nologo src\tkm.rc
        if errorlevel 1 (
            echo [Error] Resource compilation failed!
            exit /b 1
        )
        set RES_FILE=src\tkm.res
    ) else (
        echo [Tokamak] rc.exe not found; building without the optional icon resource.
        set RES_FILE=
    )
) else (
    set RES_FILE=
)

echo [Tokamak] 1/2 Building standalone executable tkm.exe...
nvcc %FLAGS% %ARCH% ^
    src\tkm_registry.c ^
    src\tkm_core.c ^
    src\tkm_codegen.c ^
    src\tkm_gpu.cu ^
    src\main.c ^
    %RES_FILE% ^
    -lnvrtc -lcuda -lcublas ^
    -o tkm.exe

if %ERRORLEVEL% neq 0 (
    echo [Error] tkm.exe build failed!
    exit /b 1
)

echo [Tokamak] 2/2 Building shared library tkm.dll (+ tkm.lib)...
nvcc --shared %FLAGS% -DTKM_BUILD_DLL %ARCH% ^
    src\tkm_registry.c ^
    src\tkm_core.c ^
    src\tkm_codegen.c ^
    src\tkm_gpu.cu ^
    -lnvrtc -lcuda -lcublas ^
    -o tkm.dll

if %ERRORLEVEL% neq 0 (
    echo [Error] tkm.dll build failed!
    exit /b 1
)

echo.
echo [Success] Built both tkm.exe and tkm.dll (+ tkm.lib)!