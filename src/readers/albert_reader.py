"""
    This is the virtual base class of retriever
"""

from utils import *
from model.modeling_albert import AlbertModel
from readers import BertReader


class AlbertReader(BertReader):

    def __init__(self, documents, config):

        super().__init__(documents, config, use_bert=False)

        # overide BertModel in BertReader
        self.bert = AlbertModel.from_pretrained(self.bert_dir)
        self.init(use_bert=False)

