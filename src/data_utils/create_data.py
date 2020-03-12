import os
import sys
import json
import numpy as np

from statistic import collect_statistic

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  + '/'
RAW_DATA_DIR = CUR_DIR + "../../data/raw_data/"
NEW_DATA_DIR = CUR_DIR + "../../data/processed_data/"
if not os.path.exists(NEW_DATA_DIR):
    os.makedirs(NEW_DATA_DIR)

SEED = 2020 # random seed to split train/val data
np.random.seed(SEED)


# 1. load documents
document = {}
with open(RAW_DATA_DIR + "context.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        e = line.strip().split('\t')
        document[e[0]] = {"context": e[1]}

# 2. load train data 
train_data = {}
with open(RAW_DATA_DIR + "train.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        e = line.strip().split('\t')
        if not e: break
        train_data[e[0]] = {"question":e[2] ,"docid":e[1], "answer":e[3]}

# 3. split train_data into val_data
ratio = 0.2
keys = list(train_data.keys())
index = np.random.choice(len(keys), int(ratio*len(keys)), replace=False)
val_keys = [list(train_data.keys())[i] for i in index]
val_data = dict([(key,train_data[key]) for key in keys if key in val_keys])
new_train_data = dict([(key,train_data[key]) for key in keys if key not in val_keys])
train_data = new_train_data

# 4. load test_data
test_data = {}
with open(RAW_DATA_DIR + "test.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        e = line.strip().split('\t')
        if not e: break
        test_data[e[0]] = {"question":e[1]}

# 5. check data
# check whether all answers in documents

# 6. process data
# include clean/segment/...

# 7. collect statistic again 
collect_statistic(train_data, "Train")
collect_statistic(val_data, "Val")
collect_statistic(test_data, "Test")
collect_statistic(document, "Document")


# 8. export data
with open(NEW_DATA_DIR + "train.json", "w") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)  # sort_keys=True
with open(NEW_DATA_DIR + "val.json", "w") as f:
    json.dump(val_data, f, indent=4, ensure_ascii=False)  # sort_keys=True
with open(NEW_DATA_DIR + "test.json", "w") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)  # sort_keys=True
with open(NEW_DATA_DIR + "document.json", "w") as f:
    json.dump(document, f, indent=4, ensure_ascii=False)  # sort_keys=True



