
# MDMRC: Multi-Document Machine Reading Comprehensive

## 0. Prepare the environment
conda create --name mdmrc python=3.6
pip install -r requirements.txt

## 1. Prepare data
cd src/data_utils
python creat_data.py

## 2. Train
cd src
python train.py

## 3. Evaluate and Generate
cd src
python train.py --forward-only
