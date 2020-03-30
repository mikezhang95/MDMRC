
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
        word = line.split('\t')[0]
        city_words[word] = [word, "city"]

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
            
for w in syn_words:
    jieba.add_word(w)
    
    
def my_tokenize(text, filter_stop_word = True, norm_flag = True):
    tokens= jieba.lcut(text)
    if filter_stop_word:  
        tokens = list(filter(lambda x: x not in stop_words, tokens))
#         if norm_flag:
#             new_tokens = []
#             for w in tokens:
#                 if w in syn_words:
#                     new_tokens.append(syn_words[w][0])
#                 else:
#                     new_tokens.append(w)
#             tokens = new_tokens
    return tokens


