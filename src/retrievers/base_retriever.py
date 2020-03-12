"""
    This is the virtual base class of retriever
"""

import torch

class BaseRetriever(torch.nn.Module):

    def __init__(self, documents, config):
        """ 
            Args:
                - documents: a big dict for all documents 
        """
        super().__init__()
        self.documents = documents
        self.config = config

    def forward(self, queries):
        """
            Args:
                - queries: batch_size * dict, dict contains the information related to this query
            Returns:
                - "candidates":[(doc_id, score),...] inserted in every query_dict
                - loss: loss if in training mode
        """
        return 0

    def update(self, loss):
        pass


    # train and validate




