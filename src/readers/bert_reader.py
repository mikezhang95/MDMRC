"""
    This is the virtual base class of retriever
"""

import torch
from torch.nn import Dropout, Linear, RReLU
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel

from readers import BaseReader
from metrics import *
from utils import *

class BertReader(BaseReader):

    def __init__(self, documents, config):
        super().__init__(documents, config)

        self.init_network()
    

        # init tokenizer
        self.bert_dir = BASE_DIR + self.config.bert_dir
        vocab_file = self.bert_dir + "vocab.txt"
        self.tokenizer = BertTokenizer(vocab_file)

        # init optimzier
        lr = self.config.lr
        self.optimizer = AdamW(self.parameters(), lr=lr, correct_bias=False)
        num_training_steps = config.num_epoch * config.num_samples / config.batch_size
        num_warmup_steps = 0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    @cost
    def read(self, queries, mode="test"):
        # 1. calculate logits
        start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels = self.forward(queries)

        # 2. predict
        self.predict(queries, start_logits, end_logits, input_seqs, query_lens)

        # 3. compute_loss
        if mode == "train":
            loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels)
        else:
            loss = 0
        return loss


    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    

    def init_network(self):
        # dropout part
        self.dropout = Dropout(self.config.dropout)

        # bert part
        bert_dir = BASE_DIR + self.config.bert_dir
        self.bert = BertModel.from_pretrained(bert_dir)
        bert_config = self.bert.config
        bert_hidden = bert_config.hidden_size # 768

        # head part
        head_hidden = self.config.head_hidden
        self.start_hidden = Linear(bert_hidden, head_hidden, bias=True)
        self.start_head = Linear(head_hidden, 1, bias=True)
        self.end_hidden = Linear(bert_hidden, head_hidden, bias=True) 
        self.end_head = Linear(head_hidden, 1, bias=True)

        # activation
        self.activation = RReLU(inplace=True)
    

    def forward(self, queries):
        # 0. clear optimizer (for training)
        self.optimizer.zero_grad()

        # 1. create inputs for BERT
        input_ids, attention_mask, token_type_ids, input_seqs, query_lens, start_labels, end_labels = self.create_input(queries)

        # batch_size = real_batch_size * candidate_num
        out_seq, _ = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        out_seq = self.dropout(out_seq) # [bs,seq_len,768]

        start_hidden = self.start_hidden(out_seq)
        start_logits = self.start_head(self.activation(start_hidden)).squeeze(-1)
        end_hidden = self.end_hidden(out_seq)
        end_logits = self.end_head(self.activation(end_hidden)).squeeze(-1)

        # start_labels, end_labels
        return start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels


    @cost
    def create_input(self, queries):
        """
            Returns:
                - input_ids
                - attention_mask
                - token_type_ids
                - input_seqs
                - query_lens
                - start_labels
                - end_labels
        """
        input_seqs = []  # 输入序列
        attention_mask = []  #  序列mask
        token_type = []  # sentence A/B
        query_lens = [] # length of [CLS] + text_a + [SEP]

        # for training only 
        start_labels, end_labels = [], []

        for query in queries:

            text_a = query["bert_cut"]

            docs = query["doc_candidates"] 
            for doc in docs: 

                doc_id, doc_score = doc

                text_b = self.documents[doc_id]["bert_cut"]
                inp_seq = ["[CLS]"] + text_a + ["[SEP]"] + text_b + ["[SEP]"]
                input_seqs.append(inp_seq)
                attention_mask.append([1] * len(inp_seq))
                token_type.append([0]*(2+len(text_a)) + [1]*(1+len(text_b)))

                query_lens.append(2 + len(text_a))
                # create label for training
                if "doc_id" in query and doc_id == query["doc_id"]:
                    # [CLS]+text_a+[SEP]+document
                    start_labels.append(query_lens[-1] + query["start_bert"])
                    end_labels.append(query_lens[-1] + query["end_bert"])
                else:
                    start_labels.append(0) # [CLS]
                    end_labels.append(0) # [CLS]

        # 1. 生成input_ids
        pad_input_seqs = pad_sequence(input_seqs, '[PAD]') # 补全query到同一个长度
        input_ids = [self.tokenizer.convert_tokens_to_ids(sq) for sq in pad_input_seqs] # 字符token转化为词汇表里的编码id
        input_ids = to_torch(np.array(input_ids),use_gpu=self.config.use_gpu)

        # 2. 生成attention_mask
        attention_mask = pad_sequence(attention_mask)
        attention_mask = to_torch(np.array(attention_mask),use_gpu=self.config.use_gpu)

        # 3. 生成token_type_ids
        token_type_ids = pad_sequence(token_type)
        token_type_ids = to_torch(np.array(token_type_ids),use_gpu=self.config.use_gpu)

        # 4. 生成labels
        start_labels = to_torch(np.stack(start_labels).reshape(-1), use_gpu=self.config.use_gpu)
        end_labels = to_torch(np.stack(end_labels).reshape(-1), use_gpu=self.config.use_gpu)
        # start_labels = to_torch(np.stack(start_labels).reshape(-1,1), use_gpu=self.config.use_gpu)
        # end_labels = to_torch(np.stack(end_labels).reshape(-1,1), use_gpu=self.config.use_gpu)

        return input_ids, attention_mask, token_type_ids, input_seqs, query_lens, start_labels, end_labels


    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        """
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        start_loss = loss_fn(start_logits, start_labels)
        end_loss = loss_fn(end_logits, end_labels)
        loss = start_loss + end_loss

        return loss


    def predict(self, queries, start_logits, end_logits, input_seqs, query_lens):
        """
            Generated keys:
                - "answer_pred"
                - "doc_id_pred"
        """
        i = 0
        for query in queries:

            num = len(query["doc_candidates"])

            start_logit = start_logits[i:i+num]
            end_logit = end_logits[i:i+num]
            input_seq = input_seqs[i:i+num]
            query_len = query_lens[i:i+num]
            i += num

            answer = find_best_answer(input_seq, query_len, start_logit, end_logit)
            query["answer_pred"] = answer[0]
            query["doc_id_pred"] = query["doc_candidates"][answer[1]][0]


# Method 1: y = max_z argmax_y[ p(y|z,x) ]
def find_best_answer(seqs, lens, start_logits, end_logits, weights=None):
    final_answer = ""
    final_doc_id = 0
    max_score = -np.inf

    # for some documents
    doc_cnt = -1
    for seq, length, start_logit, end_logit in zip(seqs, lens, start_logits, end_logits):

        doc_cnt += 1
        # for one document
        s, e = length, len(seq)-1
        i = to_numpy(torch.argmax(start_logit[s:e-1])) + s 
        j = to_numpy(torch.argmax(end_logit[i+1:e])) + i+1
        answer = "".join(seq[i:j])
        score = start_logit[i] + end_logit[j] - start_logit[0] - end_logit[0]

        # update best result
        if score > max_score:
            max_score = score
            final_answer = answer
            final_doc_id = doc_cnt

    return final_answer, final_doc_id



# Crazy Slow ...
# # Method 2: y = argmax_y [ sum_z p(y|z,x)*p(z|x) ]
# def find_best_answer(seqs, lens, start_logits, end_logits, weights):
    # """
        # Find best answer from all document candidates
        # Args:
    # """
    # spans = {}
    # for seq, length, start_logit, end_logit, weight in zip(seqs, lens, start_logits, end_logits, weights):

        # # calculate probability
        # # TODO: avoid overflow
        # start_prob = torch.exp(start_logit)/torch.sum(torch.exp(start_logit))
        # end_prob = torch.exp(end_logit)/torch.sum(torch.exp(end_logit))
        # for i in range(length, len(seq)):
            # for j in range(i,len(seq)):
                # prob = start_prob[i] * end_prob[j]
                # answer = seq[i:j+1]
                # if answer in spans:
                    # spans[answer] += prob * weight
                # else:
                    # spans[answer] = prob * weight
        # sorted_answer = sorted(spans.items()，key=lambda x: x[1])
        # return sorted_answer[-1][0], 0

