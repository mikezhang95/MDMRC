"""
    This is the virtual base class of retriever
"""

import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from readers import BaseReader


class BertReader(BaseReader):

    def __init__(self, documents, config):
        super().__init__(documents, config)

        self.init_network()
    
        # init tokenizer
        vocab_file = self.config.bert_vocab_file
        self.tokenizer = BertTokenizer(vocab_file)

        # init optimzier
        lr = self.config.lr
        self.optimizer = AdamW(self.parameters(), lr=lr, correct_biase=False)
        num_training_step = config.epoch * conifg.num_samples / config.batch_size
        num_warmup_steps = 0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    def init_network(self):
        # dropout part
        self.dropout = Dropout(self.config.dropout)

        # bert part
        bert_config_file = self.config.bert_config_file
        bert_config = BertConfig.from_pretrained(bert_config_file)
        bert_hidden = bert_config.hidden_size # 768
        self.bert = BertModel(bert_config)

        # head part
        head_hidden = self.config.head_hidden
        self.start_hidden = Linear(bert_hidden, head_hidden, activation="rrelu")
        self.start_head = Linear(head_hidden, 1)
        self.end_hidden = Linear(bert_hidden, head_hidden, activation="rrelu")
        self.end_head = Linear(head_hidden, 1)
    
    def forward(self, queries):


        input_ids, token_type_ids, attention_mask, input_seqs 
            = self.create_input(queries)

        # bs = real_batch_size * candidate_num
        out_seq, _ = self.bert(input_ids, token_type_ids=token_type_ids
                attention_mask=attention_mask)

        out_seq = self.dropout(out_seq) # [bs,seq_len,768]

        start_logits = self.start_head(self.start_hidden(out_seq)) # [bs,seq_len,2]
        end_logits = self.end_head(self.end_hidden(out_seq) )# [bs,seq_len,2]

        self.create_output(self, queries, start_logits, end_logits)


    def create_input(self, queries):
        """
            Generated keys:
                "start_label": #candidate
                "end_label": #candidate
                "input_seq": #candidate * []
            
        """

        input_seqs = []  # 输入序列
        attention_mask = []  #  序列mask
        token_type = []  # sentence A/B

        for query in queries:
            if "bert_cut_context" not in query:
                query["bert_cut_context"] = self.tokenizer(query["context"])
            text_a = query["bert_cut_context"]
            docs = query["doc_candidates"] 

            start, end = [], []
            for doc in docs: # num * seq_len

                doc_id, doc_score = docs

                # create label for
                if "doc_id" in query:
                    if doc_id == query["doc_id"]:
                        start.append(query["start"])
                        end.append(query["end"])
                    else:
                        start.append(0)
                        end.append(0)

                if "bert_cut_context" not in self.documents[doc_id]:
                    self.documents[doc_id]["bert_cut_context"] = self.tokenizer(self.documents[doc_id]["context"])

                text_b = self.documents[doc_id]["bert_cut_context"]
                inp_seq = ["[CLS]"] + text_a + ["[SEP]"] + text_b + ["[SEP]"]
                input_seqs.append(inp_seq)
                attention_mask.append([1] * len(inp_seq))
                token_type.append([0] * len(text_a) + [1] * len(text_b))

            query["start_label"] = start_label
            query["end_label"] = end_label
            
        # 1. 生成input_ids
        input_seqs = pad_sequence(input_seqs, '[PAD]') # 补全query到同一个长度
        input_ids = [self.tokenizer.convert_tokens_to_ids(sq) for sq in input_seqs] # 字符token转化为词汇表里的编码id

        # 2. 生成attention_mask
        attention_mask = pad_sequence(attention_mask)

        # 3. 生成token_type_ids
        token_type_ids = pad_sequence(token_type)

        return input_ids, attention_mask, token_type_ids, input_seq


    def create_output(self, queries, start_logits, end_logits):
        """
            Generated keys:
                - "start_logit"
                - "end_logit"
        """
        i = 0
        for query in queries:
            doc_ids = query["doc_candidates"] 
            num_doc = len(doc_ids)
            query["start_logit"] = start_logits[i:i+num_doc]
            query["end_logit"] = end_logits[i:i+num_doc]

    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        pass
