#!/usr/bin/env python3
"""Natural Language Processing - Evaluation Metrics
"""
import numpy as np


def ngram_precision(candidate_ngrams, reference_ngrams):
    """Calculate precision for n-grams"""
    clipped_counts = 0

    # Count clipped n-grams
    for ngram in candidate_ngrams:
        if ngram in reference_ngrams:
            clipped_counts += 1

    total_counts = len(candidate_ngrams)

    # Avoid division by zero
    if total_counts == 0:
        return 0

    precision = clipped_counts / total_counts
    return precision


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence"""

    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))

    candidate_ngrams = [tuple(sentence[i:i + n])
                        for i in range(len(sentence) - (n - 1))]
    reference_ngrams = [tuple(ref[i:i + n])
                        for ref in references
                        for i in range(len(ref) - (n - 1))]
    precision = ngram_precision(candidate_ngrams, reference_ngrams)

    return BP * np.exp(np.log(precision))
