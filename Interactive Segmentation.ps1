echo "--------------------------------------------------------------------------------"
echo "Interactive Segmentation App Launcher"
echo "--------------------------------------------------------------------------------"

$CondaChildItems = gci .\conda\ -erroraction 'silentlycontinue' | Out-String
if ($CondaChildItems -eq "") {
    echo "Installing Miniconda locally in the project folder..."
    echo "downloading Miniconda..."
    curl.exe --output Miniconda3-latest-Windows-x86_64.exe --url https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    mkdir conda
    $CondaDir = "$($PWD.path)\conda"
    echo "installing Miniconda..."
    Start-Process .\Miniconda3-latest-Windows-x86_64.exe -Wait -ArgumentList @("/S", "/D=$($CondaDir)")
    Remove-Item .\Miniconda3-latest-Windows-x86_64.exe
}

$CondaExecPath = ".\conda\condabin\conda.bat"
$MambaExecPath = ".\conda\condabin\mamba.bat"
$CondaEnvsDir = "$($PWD.path)\.conda"
$CondaEnvDir = "$($PWD.path)\.conda\iseg"
$CondaEnvList = Invoke-Expression "$($CondaExecPath) env list | out-string"

if (!($CondaEnvList -like "*iseg*")) {
    echo "Creating iseg environment..."
    echo "--------------------------------------------------------------------------------"
    Invoke-Expression "$($CondaExecPath) config --set remote_max_retries 3"
    Invoke-Expression "$($CondaExecPath) install -y -c conda-forge mamba" -erroraction stop
    Invoke-Expression "$($MambaExecPath) env create -p $($CondaEnvsDir)\iseg -f `
    .\environment.yml python=3.10" -erroraction stop
}

echo "Activating iseg environment..."
Invoke-Expression "$($CondaExecPath) activate $($CondaEnvDir)"

echo "Launching Interactive Segmentation App..."
python iseg.py

read-host "Press ENTER to continue..."
