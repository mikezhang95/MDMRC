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


    def retrieve(self, queries, mode="test"):
        # 1. calculate logits
        self.forward(queries)

        # 2. predict
        self.predict(queries, self.config.retriever_topk)

        # 3. compute_loss
        if mode == "train":
            return self.compute_loss(queries)
        else:
            return 0


    def compute_loss(self, queries):
        """
            Needed keys:
                - "doc_id": an int
                - "doc_logit": a torch tensor
            Returns:
                - cross_entropy loss
        """

        # 1. calculate logits
        self.forward(queries)

        # 2. calculate cross entropy loss
        logits, labels = [], []
        for query in queries:
            logit = query["doc_logit"]
            label = self.doc_list.index(query["doc_id"])
            logits.append(logit)
            labels.append(to_torch(np.array(label), use_gpu=self.config.use_gpu))

        logits = torch.stack(logits)
        labels = torch.stack(labels)
        assert labels.size()[0] == logits.size()[0]

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss


    def predict(self, queries, topk=-1):
        """
            Needed keys:
                - "doc_id": an int
            Generated keys:
                - "doc_candidates": a list of tuple (id, score) 
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

            # retrieve cheat (give groud truth to reader)
            l = [d[0] for d in selected_docs]
            if self.config.retriever_cheat and 'doc_id' in query:
                if query['doc_id'] not in d[0]:
                    max_prob = selected_docs[0][1]
                    selected_docs.insert((query['doc_id'], max_prob),0)
                    selected_docs.pop()


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
