"""
    Metrics... (precision, recall, f1, bleu4, ngram...)
"""

from collections import Counter
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
bleu_smoothing_function = SmoothingFunction().method1


def to_tokens(string):
    """
        string or list to be processed
    """
    if not isinstance(string, list):
        # TODO: jieba here or at preprocessing
        string_tokens = string.split()
    else:
        string_tokens = string
    return string_tokens




def precision_recall_f1(prediction, ground_truth):
    """
        This function calculates and returns the precision, recall and f1-score
        Args:
            - prediction: prediction 
            - ground_truth: golden string or list reference
        Returns:
            - (p, r, f1)
    """

    prediction_tokens = to_tokens(prediction)
    ground_truth_tokens = to_tokens(ground_truth)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1



def bleu4(reference, hypothesis):
    """
        This function calculates and returns the bleu-4
        Args:
            - reference: 
            - hypothesis: 
    """
    reference_tokens = to_tokens(reference)
    hypothesis_tokens = to_tokens(hypothesis)
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens,
                        smoothing_function=bleu_smoothing_function)
    return bleu_score


def topk_fn(probs, label, topk):
    """
        Input must be numpy
    """ 

    idx = list(range(len(probs)))
    idx.sort(key=lambda i: probs[i])

    assert label < len(probs)
    position = idx.index(label)

    result = []
    for k in topk:
        if position<k:
            result.append(1)
        else:
            result.append(0)

    return result




