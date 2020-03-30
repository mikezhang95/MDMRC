

import re
import numpy as np
import jieba
from gensim.summarization.bm25 import BM25
from tqdm import tqdm
import heapq
from word_util import my_tokenize, syn_words


def create_neg(train_data):
    print("="*6, " Creating Negative Candidates ", "="*6)
    for data in train_data:
        data["neg_cand"] = []
        data["neg_weight"] = []
        for doc in data["bm25_result"]:
            if doc[0] not in data["pos_cand"]:
                data["neg_cand"].append(doc[0])
                data["neg_weight"].append(doc[1])  # TODO: you can use bert result here!
        # del data["bm25_result"]


def create_pos(train_data, paragraphs, A_LEN=200, DRIFT=150):
    print("="*6, " Creating Positive Candidates ", "="*6)
    pos_count = {}
    for i in range(6):
        pos_count[i] = 0
    pos = 0
    for i, data in enumerate(train_data):
        # YZ: cut-off the long answer
        answer = data['answer'][:min(A_LEN, len(data['answer']))]
        doc_id = data['doc_id']

        data["pos_cand"] = []
        data["start"] = []
        data["end"] = []
        for i in range(500):
            paragraph_id = doc_id + '-p' + str(i)

            if paragraph_id not in paragraphs:
                break

            # use previous finding
            context = paragraphs[paragraph_id]["context"]
            start_index = i * DRIFT 
            end_index = i * DRIFT + len(context)
            for pos in data["orig_start"]:
                if pos >= start_index  and pos + len(answer) <= end_index:
                    start = pos - start_index 
                    end = start + len(answer)
                    data["pos_cand"].append(paragraph_id)
                    data['start'].append(start)
                    data['end'].append(end)
                    pos += 1


            # start_ids = find_all_indexes(context, answer)
            # if len(start_ids) > 0:
                # start = start_ids[0]
                # end = start + len(answer) # query[start:end] 
                # data["pos_cand"].append(paragraph_id)
                # data['start'].append(start)
                # data['end'].append(end)
                # pos += 1

       #  if len(data["start"]) > 10:
            # print(data["answer"])
            # print(data["pos_cand"])
            # print("\n")

        del data["orig_start"], data["orig_end"]
        if len(data["start"]) == 0:
            print("Cannot Create Find Answer in Document")
            print(data)
            raise NotImplementedError

        pos_count[int(len(data["start"]))] += 1

    print("{} Questions. {} Positive Samples. {}".format(len(train_data), pos, pos_count))


def find_all_indexes(input_str, search_str):
    """
        Find all indexes where searsh_str is a substring of input_str
    """
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        # print(i)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def correct_label(train_data, documents, aug_labels):
    """
        To check whether answer in documents
    """
    print("="*6, " Checking Data ", "="*6)

    cnt_0, cnt_2, lens = 0,0, []
    for i, data in enumerate(train_data):

        answer = data['answer']
        document = documents[data['doc_id']]["context"]

        start_ids = find_all_indexes(document, answer)
        if len(start_ids) == 0:
            # print(0, i, data['context'], answer, data['doc_id'])
            cnt_0 += 1
        elif len(start_ids) >=2 :
            # print(2, i, data['context'], answer, data['doc_id'])
            cnt_2 += 1
            # provide augmented labels
            assert data["doc_id"] == aug_labels[data["question_id"]]["doc_id"]
            data["orig_start"] = aug_labels[data["question_id"]]["start"]
            data["orig_end"] = aug_labels[data["question_id"]]["end"]


        elif len(start_ids) == 1:
            start = start_ids[0]
            end = start + len(answer)
            data["orig_start"] = [start]
            data["orig_end"] = [end]
            
        lens.append(len(answer))

    print("{} docs have no ansers.\n{} docs have more than one answers.".format(cnt_0, cnt_2))



# Use BM25 to retrieve
def bm25_retrieve(train_data, test_data, documents):

    print("="*6, " BM25 Retrieving ", "="*6)
    corpus = [] 
    doc_list = []
    print("Document Data") 
    for key, doc in tqdm(documents.items()):
        context = doc["context"]
#         jieba_cut = jieba.lcut(context)
        jieba_cut = my_tokenize(context, norm_flag=False)
        doc['jieba_cut'] = jieba_cut
        
    ### add tags (cities, organizations) ###
#     add_tags(documents)

    for key, doc in documents.items():
        jieba_cut = doc['jieba_cut']
        if "tag" in doc:
            jieba_cut = list(doc["tag"]["city"]) + list(doc["tag"]["org"])  + jieba_cut
        corpus.append(jieba_cut)
        doc_list.append(key)
        del doc["jieba_cut"], doc["tag"]
    bm25 = BM25(corpus)
    
    print("Train Data") 
    for i, doc in tqdm(enumerate(train_data)):
#         jieba_cut = jieba.lcut(doc["context"])
        jieba_cut = my_tokenize(context,norm_flag=False)
        match_scores = list(bm25.get_scores(jieba_cut))
        indexes = heapq.nlargest(100, range(len(match_scores)), match_scores.__getitem__)
        doc["bm25_result"] = [(doc_list[i], match_scores[i])  for i in indexes]

#     print("Test Data") 
#     for i, doc in tqdm(enumerate(test_data)):
# #         jieba_cut = jieba.lcut(doc["context"])
#         jieba_cut = my_tokenize(context, norm_flag=False)
#         match_scores = list(bm25.get_scores(jieba_cut))
#         indexes = heapq.nlargest(100, range(len(match_scores)), match_scores.__getitem__)
#         doc["bm25_result"] = [(doc_list[i], match_scores[i])  for i in indexes]

    calculate_topk(train_data, documents)
        
def calculate_topk(train_data, documents):

    topk = [1,5,10,20,30,40]
    result = {}
    for k in topk:
        result[k] = []
    
    for d in train_data:
        label = d["doc_id"]
        doc_order = [a[0] for a in d["bm25_result"]]
        order_list = [a[:len("230b6fc2a40937f9adf45ea97abad846")] for a in doc_order]
        if label in order_list:
            position = order_list.index(label)
        else:
            position = len(order_list) + 1
        for k in topk:
            if position < k:
                result[k].append(1)
            else:
                result[k].append(0)
    
    for k in topk:
        result[k] = np.mean(result[k])
    print(result)       
       
def add_tags(documents):
    
    tags = {}
    for key, doc in documents.items():
        real_doc_id = key[:-3]
        jieba_cut = doc['jieba_cut']
        if real_doc_id not in tags:
            tags[real_doc_id] = {"city":set(), "org": set()}
        for w in jieba_cut:
            if w in syn_words:
                tags[real_doc_id][syn_words[w][1]].add(syn_words[w][0])

    for key, doc in documents.items():
        real_doc_id = key[:-3]
        doc['tag'] = tags[real_doc_id]                          
                                                            
# Calculate Top
if __name__ == '__main__':
    
    import os
    from split_doc import split_doc
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))  + '/'
    RAW_DATA_DIR = CUR_DIR + "../../data/clean_data/"

    # 1. load documents
    document = {}
    with open(RAW_DATA_DIR + "context.csv", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            e = line.strip().split('\t')
            document[e[0]] = {"context": " ".join(e[1:])} 

    # 2. load train data 
    train_data = []
    with open(RAW_DATA_DIR + "train.csv", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            e = line.strip().split('\t')
            if not e: break
            record = {"question_id":e[0], "doc_id":e[1],
                    "context":e[2], "answer":e[3]}
            train_data.append(record)

    # 3. load test_data
    test_data = [] 
    with open(RAW_DATA_DIR + "test.csv", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            e = line.strip().split('\t')
            if not e: break
            test_data.append({"question_id":e[0], "context":e[1]})
                          
    paragraph = split_doc(document)
#     document = paragraph # document becomes the new paragraph
    document = {}
    for key, doc in paragraph.items():
        document[key] = doc
        if len(document)==50:
            break
    bm25_retrieve(train_data[:50], test_data, document)
