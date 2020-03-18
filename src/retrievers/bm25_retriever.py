"""
    Use BM25 model as retriever # not trainable  
    Implementation from https://pypi.org/project/rank-bm25/
"""
import numpy as np
import torch
from rank_bm25 import BM25Okapi

from utils import *
from retrievers import BaseRetriever

class BM25Retriever(BaseRetriever):

    def __init__(self, documents, config):

        # init base model
        super().__init__(documents,config)
        
        # create BM25 model
        self.corpus = [] 
        for key, doc in documents.items():
            self.corpus.append(doc['jieba_cut'])

        self.bm25 = BM25Okapi(self.corpus)

    # calculate (query, document) logit
    def forward(self, queries):

        logits, labels = [], []
        for query in queries:

            match_scores = list(self.bm25.get_scores(query['jieba_cut']))
            logit = to_torch(np.array(match_scores),use_gpu=self.config.use_gpu, dtype=torch.float) 
            logits.append(logit)

            if "doc_id" in query:
                label = self.doc_list.index(query["doc_id"])
                labels.append(np.array(label))
            else:
                labels.append(0)

        labels = to_torch(np.stack(labels).reshape(-1), use_gpu=self.config.use_gpu)
        logits = torch.stack(logits)

        return logits, labels


if __name__ == '__main__':

    import time
    import json

    train_data = []
    with open('../../data/processed/train.json','r') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))
            break

    with open('../../data/processed/document.json','r') as f:
        documents = json.load(f)
            
    r = BM25Retriever(documents)

    start =  time.time()
    r.forward(train_data, topk=-1)
    end = time.time()
    print("BM25 costs {}s".format(end-start))
