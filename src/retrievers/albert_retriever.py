"""
    This is the virtual base class of retriever
"""

from model.modeling_albert import AlbertModel
from retrievers import BertRetriever

class AlbertRetriever(BertRetriever):

    def __init__(self, documents, config):

        super().__init__(documents, config)
        # overide BertModel in BertReader
        self.bert = AlbertModel.from_pretrained(self.bert_dir)
