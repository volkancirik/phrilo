#!/usr/bin/env/bash

conda create -n phrilo python=3.7 --yes
conda activate phrilo

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
conda install h5py ipykernel ipywidgets tqdm scikit-image -y
conda install -c conda-forge spacy -y

pip install pytorch-pretrained-bert
python -m ipykernel install --user --name ban3.5 --display-name "PHRILO"
python -m spacy download en
python -m pip install --upgrade --user ortools
