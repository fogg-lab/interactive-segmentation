@ECHO OFF
SETLOCAL

echo Activating conda environment...
%CondaExecPath% activate %CondaEnvDir%

echo Launching Interactive Segmentation App...
python iseg.py

PAUSE
