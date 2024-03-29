"""
    Metrics... (precision, recall, f1, bleu4, ngram...)
"""

from collections import Counter
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
bleu_smoothing_function = SmoothingFunction().method1

from rouge import Rouge
rouge = Rouge()


def to_tokens(string):
    """
        string or list to be processed
    """
    if not isinstance(string, list):
        string_tokens = list(string)
    else:
        string_tokens = string
    return string_tokens

def f1_fn(prediction, ground_truth):
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

def bleu_fn(hypothesis, reference):
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

def topk_fn(order_list, label, topk):
    """
    """ 

    
    if label in order_list:
        position = order_list.index(label)
    else:
        position = len(order_list) + 1


    result = []
    for k in topk:
        if position < k:
            result.append(1)
        else:
            result.append(0)

    return result

def rouge_fn(hypothesis, reference):
    reference_tokens = " ".join(to_tokens(reference))
    hypothesis_tokens = " ".join(to_tokens(hypothesis))
    scores = rouge.get_scores(hypothesis_tokens, reference_tokens)
    return scores[0]["rouge-l"]["f"]
