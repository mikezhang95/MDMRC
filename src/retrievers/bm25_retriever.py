"""
    Use BM25 model as retriever
    Implementation from https://pypi.org/project/rank-bm25/
"""

from rank_bm25 import BM25Okapi
from retrievers import BaseRetriever

class BM25Retriever(BaseRetriever):

    def __init__(self, documents, config):

        # init base model
        super().__init__(documents,config)
        
        # create BM25 model
        self.corpus, self.lens = [], [] 
        for key, doc in documents.items():
            self.corpus.append(doc['cut_context'])
            self.lens.append(len(doc['context']))

        self.bm25 = BM25Okapi(self.corpus)

    def forward(self, queries, topk=-1):

        if topk==-1:
            topk = len(documents)

        for query in queries:
            match_score = list(self.bm25.get_scores(query['cut_context']))
            selected_docs = list(zip(list(self.documents.keys()), match_score, self.lens))
            selected_docs.sort(key=lambda x: (-x[1], x[2]))
            selected_docs = selected_docs[:topk]
            query["candidates"] = selected_docs
        return 0



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
