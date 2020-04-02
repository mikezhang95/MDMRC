"""
    Use BM25 model as retriever # not trainable  
    Implementation from https://pypi.org/project/rank-bm25/
"""
import numpy as np

from utils import *
from retrievers import BaseRetriever

NUM_POS = 2

class NegRetriever(BaseRetriever):

    def __init__(self, documents, config):

        # init base model
        super().__init__(documents,config)

    # calculate (query, document) logit
    def retrieve(self, queries, mode="test"):

        # 0.forward(already done)

        # 1. predict
        self.predict(queries, self.config.retriever_topk)

        # 2. loss
        if mode == "train":
            loss = to_torch(np.array(0), dtype=torch.float, use_gpu=self.config.use_gpu)
        else:
            loss = 0
        return loss

    @cost
    def predict(self, queries, topk=-1):
        """
            Generated keys:
                - "doc_candidates": a list of tuple (id, score) 
                - "doc_order": a list of tuple (id, score) 
        """
        if topk==-1:
            topk = 100 # TODO: hard code here


        # 2. sort documents and return topk
        for query in queries:

            bm25_result = query["bm25_result"]
            query["doc_order"] = [b[0] for b in bm25_result]

            # do this only in train/val
            if not self.config.forward_only and 'doc_id' in query:
                pos_cands = query["pos_cand"]
                neg_cands = query["neg_cand"][:topk] 
                neg_weights = query["neg_weight"][:topk] 

                # sample positive
                num_pos = min(NUM_POS, len(pos_cands))
                selected_pos = list(np.random.choice(pos_cands, size=num_pos, replace=False))

                # sample negative
                num_neg = num_pos
                # weights: uniform or importance sampling !
                weights = softmax(np.array(neg_weights))
                selected_neg = list(np.random.choice(neg_cands, size=num_neg, replace=False, p=weights))

                selected_all = selected_pos + selected_neg 
                doc_candidates = [(s,1) for s in selected_all] # ignore the probability
                query["doc_candidates"] = doc_candidates # 2*NUM_POS
            else:
                query["doc_candidates"] = bm25_result[:topk]



