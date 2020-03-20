
import jieba
from transformers import BertTokenizer

t = BertTokenizer("../../models/albert_tiny/vocab.txt")


# TODO:  
# bm25: jieba_tokenizer for better retriever
# bert: bert_tokenizer to save computation later
def clean_text(datas):

    print("="*6, " Cleaning Data ", "="*6)

    for i, key in enumerate(datas):
        if type(datas) == dict:
            data = datas[key]
        else:
            data = key
        context = data['context']
        data['jieba_cut'] = jieba.lcut(context)
        # data['bert_cut'] = t.tokenize(context)

        tok_to_orig_index = []
        orig_to_tok_index = []
        orig_tokens = list(context)
        bert_tokens = []
        for i,token in enumerate(orig_tokens):
            orig_to_tok_index.append(len(bert_tokens))
            sub_tokens = t.tokenize(token)
            for sub_token in sub_tokens:
                if sub_token != "[UNK]":
                    tok_to_orig_index.append(i)
                    bert_tokens.append(sub_token)

        assert len(orig_to_tok_index) == len(orig_tokens)
        assert len(tok_to_orig_index) == len(bert_tokens)

        # add end symbol
        orig_to_tok_index.append(orig_to_tok_index[-1]+1)
        tok_to_orig_index.append(tok_to_orig_index[-1]+1)

        data['bert_cut'] = bert_tokens
        data['orig_to_tok_index'] = orig_to_tok_index
        data['tok_to_orig_index'] = tok_to_orig_index








