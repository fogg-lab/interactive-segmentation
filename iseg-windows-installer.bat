@ECHO OFF
SETLOCAL

echo --------------------------------------------------------------------------------
echo Interactive Segmentation App Launcher
echo --------------------------------------------------------------------------------

SET CondaDir=%cd%\conda

echo Installing Miniconda locally in the project folder...
echo --------------------------------------------------------------------------------
SET Target=Miniconda3-latest-Windows-x86_64.exe
echo Downloading %Target%...
curl.exe --output %Target% https://repo.anaconda.com/miniconda/%Target%
mkdir %CondaDir%
echo Installing Miniconda into %CondaDir%\...
START /wait "" %Target% /S /D=%CondaDir%
DEL %Target%

SET CondaExecPath=conda\condabin\conda.bat
SET CondaEnvsDir=.conda
SET CondaEnvDir=%CondaEnvsDir%\iseg

echo Creating conda environment...
echo --------------------------------------------------------------------------------
CALL %CondaExecPath% config --set remote_max_retries 3
CALL %CondaExecPath% install -y -c conda-forge mamba
CALL %CondaDir%\condabin\mamba.bat env create -p %CondaEnvsDir%\iseg -f environment.yml python=3.10

echo Installation complete.

PAUSE
