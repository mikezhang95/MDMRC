
import re
import numpy as np
import jieba
from gensim.summarization.bm25 import BM25
from tqdm import tqdm

import heapq

def create_neg(train_data):
    
    print("="*6, " Creating Negative Candidates ", "="*6)
    for data in train_data:
        data["neg_cand"] = []
        data["neg_weight"] = []
        for doc in data["bm25_result"]:
            if doc[0] not in data["pos_cand"]:
                data["neg_cand"].append(doc[0])
                data["neg_weight"].append(doc[1])  # TODO: you can use bert result here!


def create_pos(train_data, paragraphs, A_LEN=200):

    print("="*6, " Creating Positive Candidates ", "="*6)
    pos_count = {}
    for i in range(20):
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

            context = paragraphs[paragraph_id]["context"]
            start_ids = find_all_indexes(context, answer)

            if len(start_ids) > 0:
                start = start_ids[0]
                end = start + len(answer) # query[start:end] 
                data["pos_cand"].append(paragraph_id)
                data['start'].append(start)
                data['end'].append(end)
                pos += 1

       #  if len(data["start"]) > 10:
            # print(data["answer"])
            # print(data["pos_cand"])
            # print("\n")

        if len(data["start"]) == 0:
            print("Cannot Create Find Answer in Document")
            print(data, document)
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


def check_data(train_data, documents):
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
        jieba_cut = jieba.lcut(context)
        corpus.append(jieba_cut)
        doc_list.append(key)
    bm25 = BM25(corpus)

    print("Train Data") 
    for i, doc in tqdm(enumerate(train_data)):
        jieba_cut = jieba.lcut(doc["context"])
        match_scores = list(bm25.get_scores(jieba_cut))
        indexes = heapq.nlargest(100, range(len(match_scores)), match_scores.__getitem__)
        doc["bm25_result"] = [(doc_list[i], match_scores[i])  for i in indexes]

    print("Test Data") 
    for i, doc in tqdm(enumerate(test_data)):
        jieba_cut = jieba.lcut(doc["context"])
        match_scores = list(bm25.get_scores(jieba_cut))
        indexes = heapq.nlargest(100, range(len(match_scores)), match_scores.__getitem__)
        doc["bm25_result"] = [(doc_list[i], match_scores[i])  for i in indexes]
