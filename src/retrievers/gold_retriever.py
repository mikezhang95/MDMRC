"""
    Use BM25 model as retriever # not trainable  
    Implementation from https://pypi.org/project/rank-bm25/
"""
import numpy as np
import torch
from rank_bm25 import BM25Okapi

from utils import *
from retrievers import BaseRetriever

class GoldRetriever(BaseRetriever):

    def __init__(self, documents, config):
        # init base model
        super().__init__(documents,config)

    # calculate (query, document) logit
    def forward(self, queries):

        logits, labels = [], []
        for query in queries:
            match_scores = [0.0] * len(self.doc_list)
            match_scores[self.doc_list.index(query["doc_id"])] = 1.0
            logit = to_torch(np.array(match_scores),use_gpu=self.config.use_gpu, dtype=torch.float) 
            logits.append(logit)

            if "doc_id" in query:
                label = self.doc_list.index(query["doc_id"])
                labels.append(to_torch(np.array(label), use_gpu=self.config.use_gpu))

        logits = torch.stack(logits)
        labels = torch.stack(labels)

        return logits, labels

