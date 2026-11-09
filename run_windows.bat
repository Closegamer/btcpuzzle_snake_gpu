@echo off
rem Windows launcher for the snake (runs it inside WSL).
rem With no arguments it builds (if needed) and starts the snake walk: puzzles 71..160.
rem CUDA is detected inside WSL; set CUDA_HOME there if your toolkit is in a custom place.
rem
rem Examples:
rem   run_windows.bat
rem   run_windows.bat --help
rem   run_windows.bat --test
rem   run_windows.bat --spread 71 --load 90
rem   run_windows.bat --freeze 71 --load 90
rem   run_windows.bat --seconds 60
rem NOTE: cmd.exe treats  > < | &  specially; keep them out of the arguments.
setlocal
set "DISTRO=Ubuntu-22.04"

if "%~1"=="" (
    echo === snake: building if needed, then the default walk over puzzles 71..160
) else (
    echo === snake %*
)

wsl -d %DISTRO% --cd "%~dp0." -- bash cuda/run.sh %*
set "RC=%ERRORLEVEL%"
echo.
if not "%SNAKE_NO_PAUSE%"=="1" pause
exit /b %RC%
