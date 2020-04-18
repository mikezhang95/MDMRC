"""
    This is the virtual base class of retriever
"""
import random
import copy
import torch
from torch.nn import Dropout, Linear, RReLU
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel

from readers import BaseReader
from metrics import *
from utils import *

Q_LEN = 50

class BertReader(BaseReader):

    def __init__(self, documents, config, use_bert=True):

        super().__init__(documents, config)

        self.bert_dir = BASE_DIR + self.config.bert_dir
        if use_bert:
            self.init(use_bert=True)
    

    def init(self, use_bert=True):

        # create bert or not(already has albert)
        if use_bert:
            self.bert = BertModel.from_pretrained(self.bert_dir)
        self.bert.pooler.dense.weight.requires_grad = False
        self.bert.pooler.dense.bias.requires_grad = False

        # init other parts/ depends on self.bert
        self.init_network()

        # init tokenizer
        vocab_file = self.bert_dir + "vocab.txt"
        self.tokenizer = BertTokenizer(vocab_file)

        # init optimzier
        lr = self.config.lr
        self.optimizer = AdamW(self.parameters(), lr=lr, correct_bias=False)
        num_training_steps = self.config.num_epoch * self.config.num_samples / self.config.batch_size / self.config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        # init cuda
        self.n_gpu = torch.cuda.device_count()
        if self.config.use_gpu and self.n_gpu > 0 :
            self.cuda()
            # fp16 support
            if self.config.fp16:
                try:
                    from apex import amp
                    self.amp = amp
                except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                self.module_list = [self.bert, self.start_head, self.end_head]
                if self.n_gpu == 1:
                    self.module_list, optimizer = self.amp.initialize(self.module_list, self.optimizer, opt_level="O2")
                else:
                    self.module_list, optimizer = self.amp.initialize(self.module_list, self.optimizer, opt_level="O1")

            # distributed training
            if self.n_gpu > 0:
                device_ids = list(range(self.n_gpu))
                self.bert = torch.nn.parallel.DistributedDataParallel(self.bert, device_ids=device_ids, output_device=device_ids[0], find_unused_parameters=True)
                self.start_head = torch.nn.parallel.DistributedDataParallel(self.start_head, device_ids=device_ids, find_unused_parameters=True)
                self.end_head = torch.nn.parallel.DistributedDataParallel(self.end_head, device_ids=device_ids, find_unused_parameters=True)


    @cost
    def read(self, queries, mode="test"):
        # 1. calculate logits
        start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels = self.forward(queries)

        # 2. compute_loss
        if mode == "train":
            loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels)
        else:
            loss = 0

        # 3. predict
        self.predict(queries, start_logits, end_logits, input_seqs, query_lens)

        return loss


    def update(self, loss, step):

        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        if self.config.fp16:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (step + 1) % self.config.gradient_accumulation_steps == 0 :
            # clip gradients
            # max_grad_norm = 1.0
            # if args.fp16:
                # torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer), max_grad_norm)
            # else:
            #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def init_network(self):
        # dropout part
        self.dropout = Dropout(self.config.dropout)

        # bert part
        bert_config = self.bert.config
        bert_hidden = bert_config.hidden_size # 768

        # head part
        # head_hidden = self.config.head_hidden
        # self.start_hidden = Linear(bert_hidden, head_hidden, bias=True)
        # self.start_head = Linear(head_hidden, 1, bias=True)
        self.start_head = Linear(bert_hidden, 1, bias=True)
        torch.nn.init.normal_(self.start_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.start_head.bias)
        # self.end_hidden = Linear(bert_hidden, head_hidden, bias=True) 
        # self.end_head = Linear(head_hidden, 1, bias=True)
        self.end_head = Linear(bert_hidden, 1, bias=True)
        torch.nn.init.zeros_(self.end_head.bias)
        torch.nn.init.normal_(self.end_head.weight, mean=0.0, std=0.02)

        # activation
        self.activation = RReLU(inplace=True)

        self.all_parameters = [self.dropout, self.bert, self.start_head, self.end_head, self.activation]
    

    def forward(self, queries):

        # 1. create inputs for BERT
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


        # do mask on sequencese
#         seq_mask = ~attention_mask.bool()
        seq_mask = ~attention_mask.byte()
        start_logits = start_logits.masked_fill(seq_mask, -1e4)
        end_logits = end_logits.masked_fill(seq_mask, -1e4)

        # start_labels, end_labels
        return start_logits, end_logits, input_seqs, query_lens, start_labels, end_labels


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

        return input_ids, attention_mask, token_type_ids, input_seqs, query_lens, start_labels, end_labels
    
    def add_noise_to_labels(self,start_positions,end_positions,ignored_index):
        bound=self.config.noise_labels_offset_bound
        for i in range(len(start_positions)):
            s=start_positions[i]
            e=end_positions[i]
            if(e-s<=bound+1):
                continue
            start_positions[i]=max(1,s+random.randint(-bound,bound))
            end_positions[i]=max(s,e+random.randint(-bound,bound))
        start_positions.clamp_(1, ignored_index) #omit [CLS]
        end_positions.clamp_(1, ignored_index) #omit [CLS]
        return start_positions,end_positions
    
    def typeloss(self,ss,es,sp,ep):
        """
        position的标签1为负样本（0处的分数越大），0为正样本
        NOTE:sp,ep所指的tensor也发生了变化（in-place）
        """
        for i in range(len(sp)):
            if(sp[i].item()!=0 or ep[i].item()!=0):
                sp[i]=ep[i]=0
            else:
                sp[i]=ep[i]=1
        ss,es=ss[:,0].float(),es[:,0].float()
        cr=torch.nn.BCEWithLogitsLoss()
        ret=cr(ss,sp.float())+cr(es,ep.float())
        ret*=self.config.typeloss_epsilon
        return ret

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        """
        """
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
    
    def grad_penalty(self,loss):
        if loss.requires_grad==False: #val
            return 0
        #train
        ori_grad=[param.grad for param in self.parameters()]
        loss.backward(retain_graph=True)
        penalty_loss=0
        module=self.bert.module if self.n_gpu>0 else self.bert
        for i,param in enumerate(module.embeddings.parameters()):
            if ori_grad[i] is None:
                penalty_loss+=torch.norm(param.grad,p='fro')
            else:
                penalty_loss+=torch.norm(param.grad-ori_grad[i],p='fro')
        penalty_loss*=self.config.gp_epsilon
        #recover
        for i,param in enumerate(self.parameters()):
            param.grad=ori_grad[i]
            
        del ori_grad,module
        return penalty_loss

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
            query_len = query_lens[i:i+num]
            input_seq = input_seqs[i:i+num]
            i += num

            # find best span according to the logit 
            span, doc_cnt, records = find_best_answer(query_len, start_logit, end_logit)

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
def find_best_answer(query_lens, start_logits, end_logits, weights=None):

    best_span = (0, 0)
    best_score = -np.inf
    best_doc = 0
    records = []
    
    # for some documents
    doc_cnt = -1
    for length, start_logit, end_logit in zip(query_lens, start_logits, end_logits):

        doc_cnt += 1
        # for one document
        s = length
#         print(f"s={s},sl={start_logit}")
        i = to_numpy(torch.argmax(start_logit[s:])) + s 
#         print(f"i={i},el={end_logit}")
        j = to_numpy(torch.argmax(end_logit[i:])) + i
        score = start_logit[i] + end_logit[j] - start_logit[0] - end_logit[0]
        span = (i-s,j-s)

        # update best result
        if score > best_score:
            best_score = score
            best_span = span 
            best_doc = doc_cnt

        records.append([str(to_numpy(start_logit[i])), str(to_numpy(end_logit[j])), str(to_numpy(start_logit[0])), str(to_numpy(end_logit[0]))])

    return best_span, best_doc, records



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

