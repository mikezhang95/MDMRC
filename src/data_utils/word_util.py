
import jieba
from tqdm import tqdm
from gensim.summarization.bm25 import BM25

stopwords_fpath = 'stopwords.txt'
with open(stopwords_fpath, 'r') as f:
    stop_words = f.read().splitlines()
    
citywords_fpath = 'cities.txt'
city_words = {}
with open(citywords_fpath, 'r') as f:
    lines = f.read().splitlines()
    for line in lines: 
        words = line.split('|')
        for word in words:
            city_words[word] = [words[0], "city"]

orgwords_fpath = 'organizations.txt'
org_words = {}
with open(orgwords_fpath, 'r') as f:
    lines = f.read().splitlines()
    for line in lines: 
        words = line.split('|')
        for word in words:
            org_words[word] = [words[0], "org"]
syn_words = org_words.copy()
syn_words.update(city_words)


class MyTokenizer(object):
    def __init__(self, syn_words):
        for w in syn_words:
            jieba.add_word(w, freq=1000)
        
        # hard code: cheat seg_words 
        jieba.suggest_freq(("税务","机关"), True)         
        for w in city_words:
#             if w[-1] == "省" or w in ["北京市","上海市","重庆市","天津市", "北京","上海","天津","重庆","苏州","广州市", "广州"]:
            jieba.suggest_freq((w,"政府"), True) 
            jieba.suggest_freq((w,"人民政府"), True) 
            jieba.suggest_freq((w,"教委"), True) 
            jieba.suggest_freq((w,"地区"), True) 
            jieba.suggest_freq((w,"开发区"), True) 
            jieba.suggest_freq((w,"经济技术开发区"), True) 
            jieba.suggest_freq((w,"海关"), True) 
            jieba.suggest_freq((w,"工业园区"), True) 
            jieba.suggest_freq((w,"市区"), True) 
            jieba.suggest_freq((w,"省委"), True)
            jieba.suggest_freq((w,"公安局"), True)
            jieba.suggest_freq((w,"教育厅"), True)
            
            
    def tokenize(self, text, filter_stop_word = True, norm_flag = True):
        tokens= jieba.lcut(text)
        tags = []
        if filter_stop_word:  
            tokens = list(filter(lambda x: x not in stop_words, tokens))
            if norm_flag:
                new_tokens = []
                for w in tokens:
                    if w in syn_words:
                        new_tokens.append(syn_words[w][0])
                        tags.append(syn_words[w][0])
                    else:
                        new_tokens.append(w)
                tokens = new_tokens
        return tokens
#         return tokens, tags



