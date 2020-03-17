
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
        data['bert_cut'] = t.tokenize(context)

