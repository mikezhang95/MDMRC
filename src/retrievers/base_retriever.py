"""
    This is the virtual base class of retriever
"""

import torch

class BaseRetriever(torch.nn.Module):

    def __init__(self):
        pass

    def forward(self, queries, documents):
        """
            Args:
                - queries: [batch_size, ?]
                - documents: [num_documents, ?]
            Returns:
                - related_doc_ids: batch_size * [?] 
        """
        pass


    # train and validate




