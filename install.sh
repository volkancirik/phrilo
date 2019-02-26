#!/usr/bin/env/bash

conda create -n phrilo3.5 python=3.5 --yes
source activate phrilo3.5

conda install cuda92 pytorch=0.4.1 -c pytorch -y
conda install h5py ipykernel ipywidgets tqdm scikit-image -y
conda install -c conda-forge spacy -y

pip install pytorch-pretrained-bert
python -m ipykernel install --user --name ban3.5 --display-name "PHRILO 3.5"
python -m spacy download en
