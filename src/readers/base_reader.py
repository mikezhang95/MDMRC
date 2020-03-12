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
                - queries: [batch_size, ?]
                - documents: [batch_size, topk, ?]
            Returns:
                - answers: [batch_size, ?]
        """
        pass

    # train and validate




