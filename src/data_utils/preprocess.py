
import jieba


# TODO:  
# 1. tfidf mode
# 2. normalize
def clean_text(datas):
    for i, key in enumerate(datas):
        if type(datas) == dict:
            data = datas[key]
        else:
            data = key
        context = data['context']
        data['cut_context'] = jieba.lcut(context)

