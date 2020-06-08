
# MDMRC: Multi-Document Machine Reading Comprehensive

This is the framework for MDMRC problem. This framework consists of two main parts, retriever and reader. Retriever selects k documents as candidates. Reader returns the most possible answers from these candidates.    

## Requirement
```bash 
conda create --name mdmrc python=3.6
pip install -r requirements.txt
```

## Preprocess data
```bash
unzip data/clean_data.zip
cd src/data_utils      
python create_data.py  
```
Preprocessing includes spliting the documents, cleaning data, tokenization, creating labels.

## Train
Prepare pytorch version pretrained model(BERT/XLNET/ALBERT) in *models/*  folder and start to finetune the model on prepared data. Please specify parameters in *configs/*  folder.     
```bash
cd src      
python train.py --config_name neg-albert_xxlarge
```
You can find training logs and saved models in *outputs/*  folder.       


## Evaluate and Generate
```
cd src      
python train.py --config_name neg-albert_xxlarge --forward-only
```
This will automatically load the best model in the trained folder and evaluate on val data and generate on test data. Results will be saved in *ouptuts/*  folder. 



