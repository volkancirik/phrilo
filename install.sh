#!/usr/bin/env/bash
set -e
conda create -n phrilo python=3.7 --yes
source `which python | sed 's/bin\/python/etc\/profile.d\/conda.sh/g'`
conda activate phrilo
conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch -y
conda install h5py ipykernel ipywidgets tqdm scikit-image -y
conda install -c conda-forge spacy -y

git clone https://github.com/airsplay/lxmert.git; cd lxmert; pip install -r requirements.txt; mkdir -p snap/pretrained; wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained; cd -

python -m ipykernel install --user --name phrilo --display-name "PhrILo"
python -m spacy download en
python -m pip install --upgrade --user ortools
pip install pytorch-pretrained-bert
