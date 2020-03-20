"""
    Use BM25 model as retriever # not trainable  
    Implementation from https://pypi.org/project/rank-bm25/
"""
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from retrievers import BaseRetriever

class TfidfRetriever(BaseRetriever):

    def __init__(self, documents, config):

        # init base model
        super().__init__(documents,config)
        
        # create TFIDF model
        self.corpus = [] 
        for key, doc in documents.items():
            self.corpus.append(" ".join(doc["jieba_context"]))
        self.tfidf = TfidfVectorizer(ngram_range=(1,1))
        self.corpus_vec = self.tfidf.fit_transform(self.corpus).transpose()

    # calculate (query, document) logit
    def forward(self, queries):

        logits, labels = [], []

        context = []
        for query in queries:
            context.append(" ".join(query["jieba_context"]))
            if "doc_id" in query:
                label = self.doc_list.index(query["doc_id"])
                labels.append(np.array(label))
            else:
                labels.append(0)

        query_vec = self.tfidf.transform(context)
        logits = to_torch((query_vec * self.corpus_vec).todense(), use_gpu=self.config.use_gpu, dtype=torch.float)
        labels = to_torch(np.stack(labels).reshape(-1), use_gpu=self.config.use_gpu)

        return logits, labels

