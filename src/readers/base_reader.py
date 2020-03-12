"""
    This is the virtual base class of retriever
"""

import torch

class BaseReader(torch.nn.Module):

    def __init__(self):
        pass

    def forward(self, queries, documents):
        """
            Args:
                - queries: batch_size * dict, dict contains the information related to this query
                - documents: a big dict for all documents 
            Returns:
                - j(ans, score),...] inserted in every query_dict
        """
        return 0


    def update(self,loss):
        pass

    # train and validate




