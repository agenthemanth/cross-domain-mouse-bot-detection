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
REM  This is the DEMO launcher, so it turns on --oracle: the CAPTCHA page's
REM  "Attack it" sims need the answer oracle. Never use it for a real deployment;
REM  run DemoServer without --oracle there (see demo\README.md, "Running safely").
REM  Extra args pass through: a port, --host <addr>, --log-trials (participant study).
java -cp "target\classes;%WEKA%;%BOUNCE%" DemoServer --oracle %*

pause
