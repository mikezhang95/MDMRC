"""
    This is the virtual base class of retriever
"""

import torch
from torch.nn import Dropout, Linear, RReLU
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel

from retrievers import BaseRetriever
from metrics import *
from utils import *

MAX_LEN=100 # TODO: hardcode

class BertRetriever(BaseRetriever):

    def __init__(self, documents, config, use_bert=True):
        super().__init__(documents, config)

        self.bert_dir = BASE_DIR + self.config.bert_dir
        if use_bert:
            self.init(use_bert=True)


    def init(self, use_bert=True):

        # create bert or not(already has albert)
        if use_bert:
            self.bert = BertModel.from_pretrained(self.bert_dir)

        # init other parts/ depends on self.bert
        self.init_network()

        # init tokenizer
        vocab_file = self.bert_dir + "vocab.txt"
        self.tokenizer = BertTokenizer(vocab_file)

        # init optimzier
        lr = self.config.lr
        self.optimizer = AdamW(self.parameters(), lr=lr, correct_bias=False)
        num_training_steps = self.config.num_epoch * self.config.num_samples / self.config.batch_size
        num_warmup_steps = 0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        # init cuda
        if self.config.use_gpu:
            self.cuda()

        # create embeddings for all documents
        self.init_doc_embedding()


    def init_network(self):
        # dropout part
        self.dropout = Dropout(self.config.dropout)

        # bert part
        bert_config = self.bert.config
        bert_hidden = bert_config.hidden_size # 768

        # head part
        query_emb = self.config.query_emb
        self.query_layer = Linear(bert_hidden, query_emb, bias=True) 
        doc_emb = self.config.doc_emb
        self.doc_layer = Linear(bert_hidden, doc_emb, bias=True) 
        # activation
        self.activation = RReLU(inplace=True)


    def create_input(self, queries, max_len=None):
        """
            Returns:
                - input_ids
                - attention_mask
                - token_type_ids
                - labels
        """
        input_seqs = []  # BERT输入序列
        attention_mask = []  #  序列mask
        token_type = []  # sentence A/B

        # for training only 
        labels = []

        for query in queries:

            text_a = query["bert_cut"]
            inp_seq = ["[CLS]"] + text_a + ["[SEP]"] 

            input_seqs.append(inp_seq)
            attention_mask.append([1] * len(inp_seq))
            token_type.append([0]*len(inp_seq))
            
            if "doc_id" in query:
                label = self.doc_list.index(query["doc_id"])
                labels.append(np.array(label))
            else:
                labels.append(0)


        # TODO: cutoff
        # 1. 生成input_ids
        pad_input_seqs = pad_sequence(input_seqs, '[PAD]', max_len=max_len) # 补全query到同一个长度
        input_ids = [self.tokenizer.convert_tokens_to_ids(sq) for sq in pad_input_seqs] # 字符token转化为词汇表里的编码id
        input_ids = to_torch(np.array(input_ids),use_gpu=self.config.use_gpu)

        # 2. 生成attention_mask
        attention_mask = pad_sequence(attention_mask, max_len=max_len)
        attention_mask = to_torch(np.array(attention_mask),use_gpu=self.config.use_gpu)

        # 3. 生成token_type_ids
        token_type_ids = pad_sequence(token_type, max_len=max_len)
        token_type_ids = to_torch(np.array(token_type_ids),use_gpu=self.config.use_gpu)

        # 4. 生成labels
        labels = to_torch(np.stack(labels).reshape(-1), use_gpu=self.config.use_gpu)

        return input_ids, attention_mask, token_type_ids, labels


    def forward(self, queries):
        # 0. clear optimizer (for training)
        self.optimizer.zero_grad()

        # 1. create inputs for BERT
        input_ids, attention_mask, token_type_ids, labels = self.create_input(queries)

        # batch_size = real_batch_size * candidate_num
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # TODO: make sure this is [CLS]
        pooled_output = self.dropout(pooled_output) # [bs,768]

        query = self.query_layer(pooled_output)  # [bs, 100]

        logits = torch.matmul(query_emb, self.doc_emb.transpose(0,1)) # [bs, ds]
        return logits, labels


    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # self.update_doc_embedding()


    def init_doc_embedding(self):
        # create bert inputs for all docs
        doc_inputs = []
        for key, doc in self.documents.items():
            i,a,t,l = self.create_input([doc], max_len=MAX_LEN)
            doc_inputs.append((i,a,t,l))
        self.doc_loader = DataLoader(doc_inputs, shuffle=False, batch_size=self.config.batch_size)

        self.update_doc_embedding()


    # This Consumes Too Much Storage...........
    @cost
    def update_doc_embedding(self):
        doc_emb = []
        for i, doc in enumerate(self.doc_loader):
            print("{}/{}".format(i, len(self.doc_loader)))
            input_ids, attention_mask, token_type_ids, _ = doc
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            token_type_ids = token_type_ids.squeeze(1)

            _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            pooled_output = self.dropout(pooled_output) # [bs,768]
            emb = self.doc_layer(pooled_output).detach()  # [bs, 100]
            self.doc_emb.append(emb)

        self.doc_emb = torch.cat(doc_emb, dim=0) # [ds, 100]
    
