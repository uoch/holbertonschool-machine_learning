#!/usr/bin/env python3
"""Natural Language Processing - Evaluation Metrics
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """cumulative_bleu = geometric mean of n-gram BLEU scores"""
    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))
    precisions = []

    for m in range(1, n + 1):
        count_clip_ngram = 0
        count_ngram = 0

        for i in range(len(sentence) - (m - 1)):
            count_clip = 0
            for reference in references:
                count_clip += sentence[i:i + m] in [reference[j:j + m]
                                                    for j in range(len(reference) - (m - 1))]
            count_clip_ngram += min(count_clip, 1)
            count_ngram += 1

        precision = count_clip_ngram / count_ngram if count_ngram > 0 else 0
        precisions.append(precision)

    return BP * np.exp(np.mean(np.log(precisions)) if any(precisions) else -np.inf)
