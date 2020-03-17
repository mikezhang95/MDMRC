"""
    This is the virtual base class of retriever
"""


import os
import torch
from utils import *
from metrics import *

class BaseReader(torch.nn.Module):

    def __init__(self, documents, config):
        super().__init__()
        self.documents = documents
        self.doc_list = list(self.documents.keys())
        self.config = config

    def load(self, path, model_id):
        """
            load {model_id}-reader from {path}
        """
        self.load_state_dict(torch.load(os.path.join(path, '{}-reader'.format(model_id))))

    def save(self, path, model_id):
        """
            save {model_id}-reader in {path}
        """
        torch.save(self.state_dict(), os.path.join(path, '{}-reader'.format(model_id)))

    def collect_metric(self, queries):
        """
            Collect metric(bleu/f1/rouge) for evaluation
            Needed Key:
                - answer
                - answer_pred
        """
        metric_result = {"bleu":[], "f1":[], "rouge":[]}

        # bleu/rouge/f1
        for query in queries:

            # tokenize problem
            answer_gt = list(query["answer"])
            answer_pred = list(query["answer_pred"])
            
            metric_result["bleu"].append(bleu_fn(answer_pred, answer_gt))
            metric_result["f1"].append(f1_fn(answer_pred, answer_gt)[2])
            metric_result["rouge"].append(rouge_fn(answer_pred, answer_gt))

        return metric_result

 

# YZ: For a reader, only needs to write this two function
    def read(self, queries, mode="test"):
        """
            Generated keys:
                - "answer_pred"
                - "doc_id_pred"
            If mode == train: should return train loss as well
        """
        return 0

    def update(self, loss):
        """
            Update parameters
        """
        pass
