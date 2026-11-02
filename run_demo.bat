@echo off
REM ============================================================
REM  Mouse-dynamics bot detector -- one-click demo launcher
REM  Double-click this file, then open http://127.0.0.1:8787/
REM ============================================================
setlocal
cd /d "%~dp0"

set "M=%USERPROFILE%\.m2\repository"
set "WEKA=%M%\nz\ac\waikato\cms\weka\weka-stable\3.8.6\weka-stable-3.8.6.jar"
set "BOUNCE=%M%\nz\ac\waikato\cms\weka\thirdparty\bounce\0.18\bounce-0.18.jar"
set "TABLESAW=%M%\tech\tablesaw\tablesaw-core\0.43.1\tablesaw-core-0.43.1.jar"

echo.
echo [1/2] Compiling Java sources...
javac -cp "%WEKA%;%BOUNCE%;%TABLESAW%" -d target\classes src\main\java\*.java
if errorlevel 1 (
  echo COMPILE FAILED -- fix the error above and try again.
  pause
  exit /b 1
)
echo      done.

echo.
echo [2/2] Starting server on http://127.0.0.1:8787/
echo      Open that URL in Chrome. Press Ctrl+C in this window to stop.
echo.
REM  pass-through args: a port, --log-trials (participant study), --oracle (probes only)
java -cp "target\classes;%WEKA%;%BOUNCE%" DemoServer %*

pause
