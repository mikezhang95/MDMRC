
import re
import math
import numpy as np

from transformers import BertTokenizer
t = BertTokenizer("../../models/albert_tiny/vocab.txt")


D_LEN = 400 # If you dont want to split doc, just set D_LEN a very big number
STRIDE = 250 # override length
A_LEN = 200 #  A_LEN + Q_LEN should < D_LEN-STRIDE


def split(train_data, documents):

    print("="*6, " Spliting Data ", "="*6)
    print("Max Document Len: {}".format(D_LEN))
    print("Stride  Len: {}".format(STRIDE))
    print("Max Answer Len: {}".format(A_LEN))

    # Split document according to D_LEN and STIRDE
    paragraphs = split_doc(documents)

    # Correct train_data labels
    correct_label(train_data, documents, paragraphs)

    # we see paragraphs as new documents
    return train_data, paragraphs


def split_doc(documents):
    paragraphs = {}
    for doc in documents:
        context = documents[doc]["context"]
        len_context = max(len(context),STRIDE+1)
        num_paragraph = math.ceil((float(len_context)-STRIDE) / (D_LEN-STRIDE)) # num=ceil[(L-S)/(D-S)]
        documents[doc]["num_paragraph"] = num_paragraph

        for i in range(num_paragraph):
            start = i*(D_LEN - STRIDE)
            end = min(start+D_LEN, len(context))
            p_context = context[start:end]
            paragraphs["%s-p%s"%(doc,str(i))] = {"context": p_context}
    return paragraphs


def correct_label(train_data, documents, paragraphs):
    """
        After split documents , correct doc_id label and generate start/end label
    """
    for i, data in enumerate(train_data):

        answer = data['answer'][:min(A_LEN, len(data['answer']))]
        doc_id = data['doc_id']
        document = documents[doc_id]
        for i in range(document['num_paragraph']):
            paragraph_id = doc_id + '-p' + str(i)
            context = paragraphs[paragraph_id]["context"]
            start_ids = find_all_indexes(context, answer)

            # TODO: we only considers the first satisfied label
            if len(start_ids)>0:
                start = start_ids[0]
                end = start + len(answer)
                data['doc_id'] = paragraph_id
                data['start']= start
                data['end'] = end
                # # create speical label for bert
                # data['start_bert'] = len(t.tokenize(context[:start]))
                # data['end_bert'] = len(t.tokenize(context[:end]))
                break
        
        if "start" not in data:
            print("Cannot Create Find Answer in Document")
            print(data, document)
            raise NotImplementedError


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


