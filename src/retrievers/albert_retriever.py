"""
    This is the virtual base class of retriever
"""

from utils import *
from model.modeling_albert import AlbertModel
from retrievers import BertRetriever

class AlbertRetriever(BertRetriever):

    def __init__(self, documents, config):

        super().__init__(documents, config, use_bert=False)

        # overide BertModel in BertReader
        self.bert_q = AlbertModel.from_pretrained(self.bert_dir)
        self.bert_d = AlbertModel.from_pretrained(self.bert_dir)
        self.init(use_bert=False)

