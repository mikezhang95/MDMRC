"""
    Albert + regression(rougeL)
"""
import torch.nn as nn
import copy
from utils import *
from metrics import *
from model.modeling_albert import AlbertModel
from readers import BertReader
import math

Q_LEN = 50

def rouge_transform(rouge):
    return rouge

class AlbertReaderReg(BertReader):

    def __init__(self, documents, config):

        super().__init__(documents, config, use_bert=False)

        # overide BertModel in BertReader
        self.bert = AlbertModel.from_pretrained(self.bert_dir)
        # bert part
        bert_config = self.bert.config
        bert_hidden = bert_config.hidden_size # 768
        #regression
        self.reg = nn.Linear(bert_hidden,1,bias=True)
        #init others
        self.init(use_bert=False)
        
    @cost
    def read(self, queries, mode="test"):
        # 1. calculate logits
        start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels,reg_logits = self.forward(queries)

        # 2. compute_loss
        if mode == "train":
            loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels)
        else:
            loss = 0

        # 3. predict
        self.predict(queries, start_logits, end_logits, input_seqs, query_lens)
        
        # 4. regress rougeL
        if mode == "train":
            loss += self.compute_reg_loss(reg_logits,queries)

        return loss
    
    def compute_reg_loss(self,reg_logits,queries):
        thebsz=reg_logits.size(0)
        reg_logits=torch.mean(reg_logits,dim=1).reshape(thebsz)
        gt_rouge=torch.empty_like(reg_logits)
        for idx,query in enumerate(queries):
            # tokenize problem
            answer_gt = list(query["answer"])
            answer_pred = list(query["answer_pred"])
            if len(query["answer"])==0:
                gt_rouge[idx]=0
            else:
                gt_rouge[idx]=rouge_transform(rouge_fn(answer_pred, answer_gt))
        loss_fn=torch.nn.MSELoss()
        loss=loss_fn(reg_logits,gt_rouge)
        return loss
    
    def forward(self, queries):

        # 1. create inputs for BERT and verifier
        input_ids, attention_mask, token_type_ids, input_seqs, query_lens, start_labels, end_labels = self.create_input(queries)

        # batch_size = real_batch_size * candidate_num
        out_seq, _ = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        out_seq = self.dropout(out_seq) # [bs,seq_len,768]
        # start_hidden = self.start_hidden(out_seq)
        # start_logits = self.start_head(self.activation(start_hidden)).squeeze(-1)
        # end_hidden = self.end_hidden(out_seq)
        # end_logits = self.end_head(self.activation(end_hidden)).squeeze(-1)
        start_logits = self.start_head(out_seq).squeeze(-1)
        end_logits = self.end_head(out_seq).squeeze(-1)
        reg_logits=self.reg(out_seq).squeeze(-1)


        # do mask on sequencese
#         seq_mask = ~attention_mask.bool()
        seq_mask = ~attention_mask.byte()
        start_logits = start_logits.masked_fill(seq_mask, -1e4) #这里实际没有mask，attention全1
        end_logits = end_logits.masked_fill(seq_mask, -1e4)

        # start_labels, end_labels
        return start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels,reg_logits
    
    
    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        """
        """
        
        #bert loss
        ignored_index=start_logits.size()[1]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_labels_n,end_labels_n=copy.deepcopy(start_labels),copy.deepcopy(end_labels)
        if self.config.add_noise_labels: #在start，end标签上加入小幅随机偏移（对抗噪声）
            start_labels_n, end_labels_n = self.add_noise_to_labels(start_labels_n, end_labels_n,ignored_index)
        start_loss = loss_fn(start_logits, start_labels_n)
        end_loss = loss_fn(end_logits, end_labels_n)
        loss = start_loss + end_loss
        if self.config.add_gp: #加入grandient penalty
            loss += self.grad_penalty(loss)
        if self.config.add_typeloss: #加入二分类损失
            loss += self.typeloss(start_logits,end_logits,start_labels_n,end_labels_n)
        return loss