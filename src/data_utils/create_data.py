import os
import sys
import json
import numpy as np
from statistic import collect_statistic
from preprocess import clean_text
from split_doc import check_data, split

# this needs to be assigned
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  + '/'
RAW_DATA_DIR = CUR_DIR + "../../data/clean_data/"
NEW_DATA_DIR = CUR_DIR + "../../data/processed/"
if not os.path.exists(NEW_DATA_DIR):
    os.makedirs(NEW_DATA_DIR)

# 1. load documents
document = {}
with open(RAW_DATA_DIR + "context.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        e = line.strip().split('\t')
        document[e[0]] = {"context": " ".join(e[1:])} # YZ: may have many '\t'

# 2. load train data 
train_data = []
with open(RAW_DATA_DIR + "train.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        e = line.strip().split('\t')
        if not e: break
        record = {"question_id":e[0], "doc_id":e[1],
                "context":e[2], "answer":e[3]}
        train_data.append(record)

# 3. load test_data
test_data = [] 
with open(RAW_DATA_DIR + "test.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        e = line.strip().split('\t')
        if not e: break
        test_data.append({"question_id":e[0], "context":e[1]})



# 5. check data
# check whether all answers in documents
# create lable like start/end
check_data(train_data, document)
train_data, document = split(train_data, document)

# 6. clean data
# include normalize/segment/...
clean_text(document)
clean_text(train_data)
clean_text(test_data)

# 7. collect statistic again 
collect_statistic(train_data, "Train")
collect_statistic(test_data, "Test")
collect_statistic(document, "Document")

# 8. export data
with open(NEW_DATA_DIR + "train.json", "w") as f:
    for data in train_data:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')
with open(NEW_DATA_DIR + "test.json", "w") as f:
    for data in test_data:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')
with open(NEW_DATA_DIR + "document.json", "w") as f:
    json.dump(document, f, indent=4, ensure_ascii=False)  # sort_keys=True



