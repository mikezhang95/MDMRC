"""
    This is the virtual base class of retriever
"""

import numpy as np
import torch
from metrics import *
from utils import *

class BaseRetriever(torch.nn.Module):

    def __init__(self, documents, config):
        """ 
            Args:
                - documents: a big dict for all documents 
        """
        super().__init__()
        self.documents = documents
        self.doc_list = list(self.documents.keys())
        self.config = config


    def compute_loss(self, queries):
        """
            Args:
                - queries: batch_size * dict, "doc_id" & "doc_logit" must be provided in dict
            Returns:
                - loss: 
        """

        # 1. calculate logits
        self.forward(queries)

        # 2. calculate cross entropy loss
        logits, labels = [], []
        for query in queries:
            logit = query["doc_logit"]
            label = self.doc_list.index(query["doc_id"])
            logits.append(logit)
            labels.append(to_torch(np.array(label)))
        logits = torch.stack(logits)
        labels = torch.stack(labels)
        assert labels.size()[0] == logits.size()[0]

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss



    def predict(self, queries, topk=-1):
        """
            Args:
                - queries: batch_size * dict, "doc_logit" must be provided in dict
            Returns:
                - add "doc_candidates" key in dict
        """
        if topk==-1:
            topk = len(self.doc_list)

        # 1. calculate logits
        self.forward(queries)

        # 2. sort documents and return topk
        for query in queries:
            match_scores = to_numpy(query["doc_logit"])
            doc_scores = list(zip(self.doc_list, match_scores))
            doc_scores.sort(key=lambda x: -x[1])
            selected_docs = doc_scores[:topk]
            query["doc_candidates"] = selected_docs


    def collect_metric(self, queries):
        metric_result = {}

        # 1. topk
        topk = [1,5,10,100]
        for k in topk:
            metric_result["top%d"%k] = []

        for query in queries:
            logit = to_numpy(query["doc_logit"])
            label = self.doc_list.index(query["doc_id"])
            result = topk_fn(logit, label, topk) 
            for i, k in enumerate(topk):
                metric_result["top%d"%k].append(result[i])

        return metric_result

    def update(self, loss):
        pass




