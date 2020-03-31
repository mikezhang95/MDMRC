
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from utils import DATA_DIR


def load_data(config):
    train_data = []
    with open(DATA_DIR + config.data_name + '/train.json','r') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))

    test_data = []
    with open(DATA_DIR + config.data_name + '/test.json','r') as f:
        for line in f.readlines():
            test_data.append(json.loads(line))

    with open(DATA_DIR + config.data_name + '/document.json','r') as f:
        documents = json.load(f)
    return train_data, test_data, documents



def get_data_loader(data, batch_size=8, split_ratio=0, shuffle=True):

    # full_dataset = Dataset(data)
    full_dataset = data

    # split dataset
    val_size = int(split_ratio*len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    # create dataloader w.r.t. dataset
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader
    

def collate_fn(batch):
    return batch



def save_badcase(metric1, metric2, data_loader, f):
    """
        Args:
            - metric1: metric for retriever
            - metric2: metric for reader
    """

    dataset = data_loader.dataset
    num_samples = len(dataset)
    
    records, topk = [], 0
    for i in range(num_samples):
        if metric2["top1"][i]==1 and metric2["rouge"][i] > 0.5:
            continue

        else: # badcase
            query = dataset[i]
            qid = query["question_id"]
            q = query["context"]
            ans = query["answer"]
            did_pred = query["doc_id_pred"] 
            ans_pred = query["answer_pred"]
            doc_order = query["doc_order"]
            pos_doc_list = query["pos_cand"] # positive label list 
            did = "||".join(pos_doc_list)

            pos  = np.inf
            for l in pos_doc_list:
                if l in doc_order:
                    p = doc_order.index(l)
                else:
                    p = 101
                if p < pos:
                    pos = p
            if pos > 100:
                pos = ">100"
            else:
                pos = str(pos)
            pos_pred = str(doc_order.index(did_pred))

            rouge = metric2["rouge"][i]

            record = [qid, q, did, ans, pos, did_pred, ans_pred, pos_pred, rouge]
            
            topk = len(query["doc_candidates"])
            for c in query["doc_candidates"]:
                record.append(c[0])
            records.append(record)

    # write to csv
    writer = csv.writer(f)
    topk_head = ["top%d"%(i+1) for i in range(topk)]
    head = ["q_id","question", "doc_id", "answer", "doc_pos", "doc_id_pred", "answer_pred", "doc_pos_pred","rouge"] + topk_head

    writer.writerow(head)
    writer.writerows(records)
    print("Saved {} badcases!".format(len(records)))
















    
    




