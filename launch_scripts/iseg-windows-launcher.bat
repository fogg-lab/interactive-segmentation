@ECHO OFF
SETLOCAL

cd ..

echo Activating conda environment...
CALL conda\condabin\conda.bat activate .conda\iseg

echo Launching Interactive Segmentation App...
python iseg.py

PAUSE
