
import json
import torch
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



def get_data_loader(data, batch_size=8, split_ratio=0):

    # full_dataset = Dataset(data)
    full_dataset = data

    # split dataset
    val_size = int(split_ratio*len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    # create dataloader w.r.t. dataset
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader
    

def collate_fn(batch):
    return batch



    
    




