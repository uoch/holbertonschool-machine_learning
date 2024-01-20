#!/usr/bin/env python3
"""Natural Language Processing - Evaluation Metrics
"""
import numpy as np


def uni_bleu(references, sentence):
    """bleu score for 1-gram"""
    candidate_len = len(sentence)
    reference_len = min(len(ref) for ref in references)

    if candidate_len > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - reference_len / candidate_len)

    count_clip_ngram = 0
    count_ngram = 0
    for word in set(sentence):
        max_match = max(ref.count(word) for ref in references)
        count_clip_ngram += min(sentence.count(word), max_match)
        count_ngram += sentence.count(word)

    if count_ngram == 0:
        return 0

    pn = count_clip_ngram / count_ngram

    return bp * pn
