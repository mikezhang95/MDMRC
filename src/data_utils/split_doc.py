
import math

def split_doc(documents, D_LEN=400, STRIDE=250):
    print("="*6, " Spliting Data ", "="*6)
    print("Max Document Len: {}".format(D_LEN))
    print("Stride  Len: {}".format(STRIDE))
    # print("Max Answer Len: {}".format(A_LEN))

    paragraphs = {}
    num_p = []
    for doc in documents:
        context = documents[doc]["context"]
        len_context = max(len(context),STRIDE+1)
        num_paragraph = math.ceil((float(len_context)-STRIDE) / (D_LEN-STRIDE)) # num=ceil[(L-S)/(D-S)]
        documents[doc]["num_paragraph"] = num_paragraph
        num_p.append(num_paragraph)

        for i in range(num_paragraph):
            start = i*(D_LEN - STRIDE)
            end = min(start+D_LEN, len(context))
            p_context = context[start:end]
            paragraphs["%s-p%s"%(doc,str(i))] = {"context": p_context}

    print("# Original Document: {}".format(len(documents)))
    print("# Split Document: {}".format(len(paragraphs)))
    print("Max Paragraphs {} Min Paragraphs {}".format(max(num_p), min(num_p)))
    print("-"*24)
    return paragraphs


