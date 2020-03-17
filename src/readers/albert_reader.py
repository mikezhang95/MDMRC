"""
    This is the virtual base class of retriever
"""

import torch
from torch.nn import Dropout, Linear, RReLU
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup 
from model.modeling_albert import AlbertModel

from readers import BertReader
from metrics import *
from utils import *

class AlbertReader(BertReader):

    def __init__(self, documents, config):

        super().__init__(documents, config)
        # overide BertModel in BertReader
        self.bert = AlbertModel.from_pretrained(self.bert_dir)
