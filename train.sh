
cd data
mkdir clean_data
unzip clean_data.zip -d clean_data

## 1. Prepare data
cd ../src/data_utils
python creat_data.py

## 2. Train
cd .. 
python train.py --config_name gold-albert_xxlarge
