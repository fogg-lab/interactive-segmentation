#!/bin/bash

echo "Downloading Miniconda..."
curl --output miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "Installing Miniconda..."
chmod +x miniconda.sh
bash miniconda.sh -b -p $PWD/../conda
rm miniconda.sh

echo "Creating conda environment..."
../conda/condabin/conda install -y -c conda-forge mamba
../conda/condabin/mamba env create -p ../conda/iseg -f ../environment.yml python=3.10
../conda/condabin/conda init bash

echo "Downloading model checkpoint file..."
rm ../checkpoints/last_checkpoint.pth
curl --output ../checkpoints/last_checkpoint.pth https://media.githubusercontent.com/media/fogg-lab/interactive-segmentation/distribution/checkpoints/last_checkpoint.pth

read -p "Installation complete. Press enter to continue"
