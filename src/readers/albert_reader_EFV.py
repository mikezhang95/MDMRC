"""
    Albert + external front verifier
"""
import torch.nn as nn
import copy
from utils import *
from model.modeling_albert import AlbertModel
from readers import BertReader

Q_LEN = 50

class AlbertReaderEFV(BertReader):

    def __init__(self, documents, config):

        super().__init__(documents, config, use_bert=False)

        # overide BertModel in BertReader
        self.bert = AlbertModel.from_pretrained(self.bert_dir)
        # bert part
        bert_config = self.bert.config
        bert_hidden = bert_config.hidden_size # 768
        #EFV
        self.efv = nn.Linear(bert_hidden,1,bias=True)
        #init others
        self.init(use_bert=False)
        
    @cost
    def read(self, queries, mode="test"):
        # 1. calculate logits
        start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels,efv_logits,efv_labels = self.forward(queries)

        # 2. compute_loss
        if mode == "train":
            loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels,efv_logits,efv_labels)
        else:
            loss = 0

        # 3. predict
        self.predict(queries, start_logits, end_logits, input_seqs, query_lens,efv_logits)

        return loss
    
    def forward(self, queries):

        # 1. create inputs for BERT and verifier
        input_ids, attention_mask, token_type_ids, input_seqs, query_lens, start_labels, end_labels, efv_labels = self.create_input(queries)

        # batch_size = real_batch_size * candidate_num
        out_seq, _ = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        out_seq = self.dropout(out_seq) # [bs,seq_len,768]
        # start_hidden = self.start_hidden(out_seq)
        # start_logits = self.start_head(self.activation(start_hidden)).squeeze(-1)
        # end_hidden = self.end_hidden(out_seq)
        # end_logits = self.end_head(self.activation(end_hidden)).squeeze(-1)
        start_logits = self.start_head(out_seq).squeeze(-1)
        end_logits = self.end_head(out_seq).squeeze(-1)
        efv_logits=self.efv(out_seq).squeeze(-1)


        # do mask on sequencese
#         seq_mask = ~attention_mask.bool()
        seq_mask = ~attention_mask.byte()
        start_logits = start_logits.masked_fill(seq_mask, -1e4) #这里实际没有mask，attention全1
        end_logits = end_logits.masked_fill(seq_mask, -1e4)

        # start_labels, end_labels
        return start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels,efv_logits,efv_labels
    
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
        input_seqs = []  # bert input sequence 
        attention_mask = []  #  attention mask
        token_type = []  # sentence A/B
        query_lens = [] # length of [CLS] + text_a + [SEP]

        # for training only 
        start_labels, end_labels = [], []

        for query in queries:
            text_a = query["bert_cut"]
            if len(text_a) > Q_LEN :
                text_a = text_a[:Q_LEN]

            for candidate in query["doc_candidates"] :

                doc_id, doc_score = candidate
                doc = self.documents[doc_id]
                text_b = doc["bert_cut"]

                # bert inputs
                inp_seq = ["[CLS]"] + text_a + ["[SEP]"] + text_b + ["[SEP]"]
                input_seqs.append(inp_seq)
                attention_mask.append([1] * len(inp_seq))
                token_type.append([0]*(2+len(text_a)) + [1]*(1+len(text_b)))

                # for predict
                orig_to_tok_index = doc["orig_to_tok_index"]
                query_lens.append(2 + len(text_a))

                # create label for training
                if "pos_cand" in query and doc_id in query["pos_cand"]:
                    i = query["pos_cand"].index(doc_id)
                    # start_labels.append(query_lens[-1] + query["start_bert"])
                    # end_labels.append(query_lens[-1] + query["end_bert"])
                    try:
                        start_labels.append(query_lens[-1] + orig_to_tok_index[query["start"][i]])
                        end_labels.append(query_lens[-1] + orig_to_tok_index[query["end"][i]])
                    except:
                        print(len(doc["context"]),len(doc["bert_cut"]))
                        print(query["start"], query["end"], doc)
                        raise NotImplementedError

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
        # 5. 生成efv_labels
        efv_labels=copy.deepcopy(start_labels)
        for i,label in enumerate(efv_labels):
            if start_labels[i].item()==0==end_labels[i].item():
                efv_labels[i]=0.
            else:
                efv_labels[i]=1.
        efv_labels=efv_labels.float()

        return input_ids, attention_mask, token_type_ids, input_seqs, query_lens, start_labels, end_labels,efv_labels
    
    def compute_loss(self, start_logits, end_logits, start_labels, end_labels,efv_logits,efv_labels):
        """
        """
        loss=0
        #EFV loss
        efv_loss_fn=torch.nn.BCEWithLogitsLoss()
        efv_logits=torch.mean(efv_logits,dim=1).reshape(efv_labels.size()[0])
        efv_loss=efv_loss_fn(efv_logits,efv_labels)
        loss+=efv_loss
        
        #bert loss
        ignored_index=start_logits.size()[1]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_labels_n,end_labels_n=copy.deepcopy(start_labels),copy.deepcopy(end_labels)
        if self.config.add_noise_labels: #在start，end标签上加入小幅随机偏移（对抗噪声）
            start_labels_n, end_labels_n = self.add_noise_to_labels(start_labels_n, end_labels_n,ignored_index)
        start_loss = loss_fn(start_logits, start_labels_n)
        end_loss = loss_fn(end_logits, end_labels_n)
        loss += start_loss + end_loss
        if self.config.add_gp: #加入grandient penalty
            loss += self.grad_penalty(loss)
        if self.config.add_typeloss: #加入二分类损失
            loss += self.typeloss(start_logits,end_logits,start_labels_n,end_labels_n)
        return loss
    
    def predict(self, queries, start_logits, end_logits, input_seqs, query_lens,efv_logits):
        """
            Generated keys:
                - "answer_pred"
                - "doc_id_pred"
        """
        efv_logits=torch.mean(efv_logits,dim=1).reshape(start_logits.size()[0])

        i = 0
        for query in queries:

            num = len(query["doc_candidates"])
            start_logit = start_logits[i:i+num]
            end_logit = end_logits[i:i+num]
            query_len = query_lens[i:i+num]
            input_seq = input_seqs[i:i+num]
            efv_logit=efv_logits[i:i+num]
            i += num

            # find best span according to the logit 
            span, doc_cnt, records = find_best_answer(query_len, start_logit, end_logit,efv_logit,self.config.lambda1,self.config.lambda2,self.config.tao)

            doc_id = query["doc_candidates"][doc_cnt][0]
            doc = self.documents[doc_id]
            tok_to_orig_index = doc["tok_to_orig_index"]
            orig_seq = doc["context"]

            query["doc_id_pred"] = doc_id
            if span[0] == span[1]:
                query["answer_pred"] = "null"
            else:
                query["answer_pred"] = orig_seq[tok_to_orig_index[span[0]]:tok_to_orig_index[span[1]]]

            # try:
                # query["answer_pred"] = orig_seq[tok_to_orig_index[span[0]]:tok_to_orig_index[span[1]]]
            # except:
                # print(span)
                # print(len(orig_seq),orig_seq)
                # print(len(input_seq[doc_cnt]),input_seq[doc_cnt])
                # print(query_len[doc_cnt])
                # print(len(tok_to_orig_index))
                # print(tok_to_orig_index[-1])
                # raise NotImplementedError


#             qid = query["question_id"]
            # for ii, record in enumerate(records):
                # did = query["doc_candidates"][ii][0]
                # logit = "\t".join(record)
                # wf.write("{}\t{}\t{}\n".format(qid, did, logit))
            
# wf = open("bert_logits.txt", "a")


# Method 1: y = max_z argmax_y[ p(y|z,x) ]
def find_best_answer(query_lens, start_logits, end_logits,efv_logits,lambda1,lambda2,tao, weights=None):

    best_span = (0, 0)
    best_score = -np.inf
    best_doc = 0
    records = []
    
    # for some documents
    doc_cnt = -1
    for length, start_logit, end_logit,efv_logit in zip(query_lens, start_logits, end_logits,efv_logits):

        doc_cnt += 1
        # for one document
        s = length
        i = to_numpy(torch.argmax(start_logit[s:])) + s 
        j = to_numpy(torch.argmax(end_logit[i:])) + i
        score_has = start_logit[i] + end_logit[j]
        score_na= lambda1*(start_logit[0]+end_logit[0])+lambda2*efv_logit
        score=score_has-score_na
        if score<tao:
            i=j=s
        span = (i-s,j-s)

        # update best result
        if score > best_score:
            best_score = score
            best_span = span 
            best_doc = doc_cnt

        records.append([str(to_numpy(start_logit[i])), str(to_numpy(end_logit[j])), str(to_numpy(start_logit[0])), str(to_numpy(end_logit[0]))])

    return best_span, best_doc, records