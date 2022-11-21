@ECHO OFF
SETLOCAL

echo Activating conda environment...
..\conda\condabin\conda.bat activate ..\.conda\iseg

echo Launching Interactive Segmentation App...
..\conda\iseg\bin\python.exe ..\iseg.py

PAUSE
