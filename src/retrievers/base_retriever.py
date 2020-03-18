"""
    This is the virtual base class of retriever
"""

import os
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


    @cost
    def retrieve(self, queries, mode="test"):

        # 1. calculate logits
        doc_logits, doc_labels = self.forward(queries)

        # 2. predict
        self.predict(queries, doc_logits, self.config.retriever_topk)

        # 3. compute_loss
        if mode == "train":
            loss = self.compute_loss(doc_logits, doc_labels)
        else:
            loss = 0

        return loss


    def compute_loss(self, logits, labels):
        """
                - "doc_logit": a torch tensor
            Returns:
                - cross_entropy loss
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss


    def predict(self, queries, doc_logits, topk=-1):
        """
            Generated keys:
                - "doc_candidates": a list of tuple (id, score) 
                - "doc_order": a list of tuple (id, score) 
        """
        if topk==-1:
            topk = len(self.doc_list)

        doc_logits = to_numpy(doc_logits)

        # 2. sort documents and return topk
        for query, doc_logit in zip(queries, doc_logits):
            doc_scores = list(zip(self.doc_list, doc_logit))
            doc_scores.sort(key=lambda x: -x[1])
            doc_order = [x[0] for x in doc_scores]

            # retrieve cheat (give groud truth to reader)
            # this only influences doc_candidates passed to reader but not doc_order.
            if self.config.retriever_cheat and 'doc_id' in query:
                p = doc_order.index(query['doc_id'])
                if p >= topk:
                    e = doc_scores.pop(p)
                    doc_scores.insert(0, e)

            selected_docs = doc_scores[:topk]
            query["doc_candidates"] = selected_docs
            query["doc_order"] = doc_order


    def collect_metric(self, queries):
        metric_result = {}

        # 1. topk
        topk = [1,5,10,100]
        for k in topk:
            metric_result["top%d"%k] = []

        for query in queries:
            doc_order = query["doc_order"]
            label = query["doc_id"]
            result = topk_fn(doc_order, label, topk) 
            for i, k in enumerate(topk):
                metric_result["top%d"%k].append(result[i])

        return metric_result


    def load(self, path, model_id):
        """
            load {model_id}-retriever from {path}
        """
        self.load_state_dict(torch.load(os.path.join(path, '{}-retriever'.format(model_id))))


    def save(self, path, model_id):
        """
            save {model_id}-retrieverin {path}
        """
        torch.save(self.state_dict(), os.path.join(path, '{}-retriever'.format(model_id)))


# YZ: For a retriever, only needs to write this two function
    def forward(self, queries):
        """
            Generated_keys:
                - "doc_logit": logit for every doc
        """ 
        pass

    def update(self, loss):
        """
            Update parameters
        """
        pass
