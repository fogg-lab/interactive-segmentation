#!/bin/bash

echo "Downloading Miniconda..."
curl --output miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

echo "Installing Miniconda..."
chmod +x miniconda.sh
bash miniconda.sh -b -p $PWD/conda
rm miniconda.sh

echo "Creating conda environment..."
./conda/condabin/conda install -y -c conda-forge mamba
./conda/condabin/mamba env create -p ./conda/iseg -f environment.yml python=3.10

read -p "Press enter to continue"
