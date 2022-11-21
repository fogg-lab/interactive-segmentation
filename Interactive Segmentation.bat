echo --------------------------------------------------------------------------------
echo Interactive Segmentation App Launcher
echo --------------------------------------------------------------------------------

SET CondaDir=%cd%\conda

if NOT exist conda\ (
    echo Installing Miniconda locally in the project folder...
    echo --------------------------------------------------------------------------------
    SET Target=Miniconda3-latest-Windows-x86_64.exe
    echo   downloading %target%...
    curl.exe --output %Target% --url https://repo.anaconda.com/miniconda/%Target%
    mkdir %CondaDir%
    echo   installing Miniconda into %CondaDir%\...
    .\%Target% /S /D=%CondaDir%
    DEL .\%Target%
)

SET CondaExecPath=%CondaDir%\condabin\conda.bat
SET CondaEnvsDir=%cd%\.conda
SET CondaEnvDir=%CondaEnvsDir%\iseg

echo Activating conda environment (if it exists)...
SET CondaEnvActivateCmd=%CondaExecPath% activate %CondaEnvDir%

echo debug_1
:try_activate_env
echo debug try_activate_env
SET EnvActivationFailed=0
%CondaEnvActivateCmd% | find "Could not find conda environment" > nul && SET EnvActivationFailed=1
EXIT /B 0

echo debug_2
CALL :try_activate_env

if %EnvActivationFailed% == 1 (
    echo Creating conda environment...
    echo --------------------------------------------------------------------------------
    %CondaExecPath% config --set remote_max_retries 3
    %CondaExecPath% install -y -c conda-forge mamba
    %CondaDir%\condabin\mamba.bat env create -p %CondaEnvsDir%\iseg -f .\environment.yml python=3.10
    echo Activating conda environment...
    echo debug_3
    CALL :try_activate_env
)

if %EnvActivationFailed% == 1 (
    echo Failed to activate conda environment!
    EXIT /B 1
)

echo Activated conda environment.

echo Launching Interactive Segmentation App...
python iseg.py

read-host "Press ENTER to continue..."
