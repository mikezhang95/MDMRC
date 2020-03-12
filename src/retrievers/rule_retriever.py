"""
    Merge all rule based methods
    Like bleu/f1/bm25

"""

import time
import json

from retrievers import BaseRetriever

class RuleRetriever(BaseRetriever):

    def __init__(self, documents, config):
        # init base model
        super().__init__(documents,config)

    def forward(self, queries, topk=-1):

        return 0
