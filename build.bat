@echo off
setlocal

if defined INCLUDE goto :msvc_ready

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%i"
)

if not defined VS_INSTALL (
    echo [Error] MSVC environment is not initialized and Visual Studio C++ tools were not found.
    echo         Run this script from an x64 Native Tools Command Prompt for VS.
    exit /b 1
)

echo [Tokamak] Initializing MSVC x64 build environment...
call "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [Error] Failed to initialize the MSVC x64 build environment.
    exit /b 1
)

:msvc_ready
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

echo [Tokamak] 1/3 Building standalone executable tkm.exe...
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

echo [Tokamak] 2/3 Building shared library tkm.dll (+ tkm.lib)...
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

echo [Tokamak] 3/3 Building MNIST IDX C ABI example mnist_idx.exe...
cl.exe /nologo /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /DTKM_USE_DLL /Iinclude ^
    examples\mnist_idx.c tkm.lib ^
    /Fe:mnist_idx.exe

if %ERRORLEVEL% neq 0 (
    echo [Error] mnist_idx.exe build failed!
    exit /b 1
)

echo.
echo [Success] Built tkm.exe, tkm.dll (+ tkm.lib), and mnist_idx.exe!