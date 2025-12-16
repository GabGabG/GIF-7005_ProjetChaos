@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=..

echo Rendering Lorenz Attractor animation...
manim render --resolution 1920,1440 --fps 30 animation.py LorenzAttractor

if exist "C:\temp\manim\videos\animation\1440p30\LorenzAttractor.mp4" (
    if not exist "media\videos\animation\1440p30" mkdir "media\videos\animation\1440p30"
    copy "C:\temp\manim\videos\animation\1440p30\LorenzAttractor.mp4" "media\videos\animation\1440p30\LorenzAttractor.mp4" >nul
    
    echo Done! Video saved to: media\videos\animation\1440p30\LorenzAttractor.mp4
    echo Original also at: C:\temp\manim\videos\animation\1440p30\LorenzAttractor.mp4
) else (
    echo Error: Video file not found!
)