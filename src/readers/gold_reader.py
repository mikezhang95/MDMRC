"""
    This is the virtual base class of retriever
"""

import torch
from readers import BaseReader
from utils import *


class GoldReader(BaseReader):

    def __init__(self, documents, config):

        super().__init__(documents, config)


    @cost
    def read(self, queries, mode="test"):
        for query in queries:
            doc_id = query["doc_candidates"][0][0]
            query["doc_id_pred"] = doc_id
            if doc_id == query["doc_id"]:
                query["answer_pred"] = query["answer"]
            else:
                query["answer_pred"] = ""

        loss = to_torch(np.array(0), use_gpu=self.config.use_gpu, dtype=torch.float)
        return loss

