#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('tokenizer_tf2_qa')
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
model = hub.load(
    "https://www.kaggle.com/models/seesee/bert/frameworks/"+\
    "TensorFlow2/variations/uncased-tf2-qa/versions/1")


def question_answer(question, reference):
    """using `[1:]` will enforce an answer. 
    `outputs[0][0][0]` is the ignored '[CLS]' token logit"""
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)
    # combine question and paragraph tokens
    tokens = ['[CLS]'] + question_tokens + \
        ['[SEP]'] + paragraph_tokens + ['[SEP]']
    # convert tokens to tensor of ids
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    # create input mask is 1 for real tokens and 0 for padding tokens
    input_mask = [1] * len(input_word_ids)
    # segment input to 0 for question tokens and 1 for paragraph tokens
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + \
        [1] * (len(paragraph_tokens) + 1)
    # prepare inputs for model but with batch size 1
    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    # short_start is the index of the first token of the answer
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    # short_end is the index of the last token of the answer
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    # build answer from tokens
    answer_tokens = tokens[short_start: short_end + 1]
    # convert tokens to string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
