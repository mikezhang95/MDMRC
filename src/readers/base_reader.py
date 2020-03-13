"""
    This is the virtual base class of retriever
    Suppose we all use Bert style inputs [CLS]text_a[SEP]text_b[SEP]
"""


import torch

class BaseReader(torch.nn.Module):

    def __init__(self):
        pass

    def forward(self, queries):
        """
            Generated_keys:
                - "start_logit": #candidate * []
                - "end_logit": #candidate * []
        """ 


    # TODO: when to insert when to insert when to insert when to insert
    """
        start_label
        end_label
        input_seq
        input_mask
    """


    def compute_loss(self, queries):
        """
            Needed keys:
                - "start_logit": #candidate * []
                - "end_logit": #candidate * []
                - "start_label": #candidate
                - "end_label": #candidate
        """

        # 1. calculate logits
        self.forward(queries)
    
        # 2. stack all logits and labels
        start_logits, end_logits = [], []
        for query in queries:
            start_logits.append(query["start_logit"])
            end_logits.append(query["end_logit"])
        start_logits = torch.stack(start_logits)
        end_logits = torch.stack(end_logits)
        start_labels = torch.stack(start_labels)
        end_labels = torch.stack(end_labels)

        # 3. calculate cross entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        start_loss = loss_fn(start_logits, start_labels)
        end_loss = loss_fn(end_logits, end_labels)

        loss = start_loss + end_loss
        return loss


    def predict(self, queries):
        """
            Needed keys:
                - "start_logit": #candidate * []
                - "end_logit": #candidate * []
                - "input_seq":
            Generated keys:
                - "answer_pred"
                - "doc_id_pred"
        """
        # 1. calculate logits
        self.forward(queries)

        # 2. sort documents and return topk
        for query in queries:
            start_logit = query["start_logit"] # num * seq_len
            end_logit = query["end_logit"]
            seqs = query["bert_tokens"]
            answer = find_best_answer(seqs, start_logit, end_logit)
            query["answer_pred"] = answer[0]
            query["doc_id_pred"] = query["doc_candidates"][answer[1]][0]


    def collect_metric(self, queries):
        metric_result = {}

        # bleu/rouge/f1
        for query in queries:

            # tokenize problem
            answer_gt = list(query["answer"])
            answer_pred = list(query["answer_pred"])
            
            metric_result["bleu"] = bleu_fn(answer_pred, answer_gt)
            metric_result["f1"] = f1_fn(answer_pred, answer_gt)[2]
            metric_result["rouge"] = rouge_fn(answer_pred, answer_gt)

        return metric_result

    def load(self, path, model_id):
        """
            load {model_id}-retriever from {path}
        """
        self.load_state_dict(torch.load(os.path.join(path, '{}-retriever'.format(model_id))))

    def save(self, path, model_id):
        """
            save {model_id}-retrieverin {path}
        """
        torch.save(self.state_dict(), os.path.join(path, '{}-retriever'.format(model_id)))

    def update(self, loss):
        pass

# # Method 1: y = argmax_y [ sum_z p(y|z,x)*p(z|x) ]
# def find_best_answer(seqs, start_logits, end_logits, weights):
    # """
        # Find best answer from all document candidates
        # Args:
    # """
    # spans = {}
    # for seq, start_logit, end_logit, weight in zip(seqs, start_logits, end_logits, weights):

        # # calculate probability
        # # TODO: avoid overflow
        # start_prob = torch.exp(start_logit)/torch.sum(torch.exp(start_logit))
        # end_prob = torch.exp(end_logit)/torch.sum(torch.exp(end_logit))
        # for i in range(len(seq)):
            # for j in range(i,len(seq)):
                # prob = start_prob[i] * end_prob[j]
                # answer = seq[i:j+1]
                # if answer in spans:
                    # spans[answer] += prob * weight
                # else:
                    # spans[answer] = prob * weight
        # sorted_answer = sorted(spans.items()，key=lambda x: x[1])
        # return sorted_answer[-1][0], 0



# Method 2: y = max_z argmax_y[ p(y|z,x) ]
def find_best_answer(seqs, start_logits, end_logits, weights=None):
    final_answer = ""
    final_doc_id = 0
    max_prob = -np.inf

    # for some documents
    doc_cnt = -1
    for seq, start_logit, end_logit in zip(seqs, start_logits, end_logits):

        doc_cnt += 1
        # for one document
        span, max_score = (0,0), -np.inf
        for i in range(len(seq)):
            for j in range(i,len(seq)):
                score = start_logit[i] + end_logit[j]
                    - start_logit[0] - end_logit[0]
                if score > max_score:
                    max_score = score
                    span = (i, j)
        answer = seq[span[0]:span[1]+1]
        prob = max_score
        if prob > max_prob:
            max_prob = prob
            final_answer = answer
            final_doc_id = doc_cnt
    
    return final_answer, final_doc_id


            






















