
import jieba


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
        data['jieba_context'] = jieba.lcut(context)

