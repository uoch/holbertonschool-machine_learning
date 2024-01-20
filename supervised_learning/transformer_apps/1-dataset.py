#!/usr/bin/env python3
"""transformer applications"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""

    def __init__(self):
        """constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        subword = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = subword((en.numpy() for _, en in data), 2**15)
        tokenizer_pt = subword((pt.numpy() for pt, _ in data), 2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encode translation pair into tokens"""
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size+1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size+1]
        return pt_tokens, en_tokens
