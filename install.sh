#!/usr/bin/env/bash
set -e
conda create -n phrilo python=3.7 --yes
source `which python | sed 's/bin\/python/etc\/profile.d\/conda.sh/g'`
conda activate phrilo
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
conda install h5py ipykernel ipywidgets tqdm scikit-image -y
conda install -c conda-forge spacy -y

python -m ipykernel install --user --name phrilo --display-name "PhrILo"
python -m spacy download en
python -m pip install --upgrade --user ortools
pip install pytorch-pretrained-bert
