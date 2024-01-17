#!/usr/bin/env python3
"""transformer applications"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""
    def update_dataset(self, data_train, data_valid, batch_size,
                       max_len):
        """update data_train and data_valid attributes"""
        self.data_train = data_train.shuffle(2048)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(
            lambda x, y: tf.logical_and(tf.size(x) <= max_len,
                                        tf.size(y) <= max_len))
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(
            lambda x, y: tf.logical_and(tf.size(x) <= max_len,
                                        tf.size(y) <= max_len))
        self.data_valid = self.data_valid.padded_batch(batch_size)
        return self.data_train, self.data_valid
    def __init__(self, batch_size, max_len):
        """constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train, self.data_valid = self.update_dataset(
            self.data_train, self.data_valid, self.batch_size, self.max_len)

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

    def tf_encode(self, pt, en):
        """tensorflow wrapper for the encode instance method"""
        pt_lang, en_lang = tf.py_function(func=self.encode,
                                          inp=[pt, en],
                                          Tout=[tf.int64, tf.int64])
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
