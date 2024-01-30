#!/usr/bin/python3
""" console """

import sys
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
model = hub.load(
    "https://www.kaggle.com/models/seesee/bert/frameworks/" +
    "TensorFlow2/variations/uncased-tf2-qa/versions/1")


def question_answer(question, reference):
    """using `[1:]` will enforce an answer. 
    `outputs[0][0][0]` is the ignored '[CLS]' token logit"""
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + question_tokens + \
        ['[SEP]'] + paragraph_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + \
        [1] * (len(paragraph_tokens) + 1)
    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer


def semantic_search(corpus_path, sentence):
    """semantic search"""
    corpus = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, 'r') as md_file:
                corpus.append(md_file.read() + '\n')

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_sen = model.encode(sentence)

    similarities = [cosine_similarity(
        [embeddings_sen], [model.encode(doc)]) for doc in corpus]
    most_similar_idx = np.argmax(similarities)
    most_similar_doc = corpus[most_similar_idx]
    return most_similar_doc


def answer_loop(coprus_path):
    """loop to answer questions from a reference text"""
    while True:
        try:
            sentence = input("Q: ")
            sentence = sentence.lower()
            if sentence in ['exit', 'quit', 'goodbye', 'bye']:
                print("A: Goodbye")
                sys.exit()
            else:
                reference = semantic_search(coprus_path, sentence)
                if question_answer(sentence, reference) == None or question_answer(sentence, reference) == "":
                    print("A: Sorry, I do not understand your question.")
                else:
                    print("A: {}".format(question_answer(sentence, reference)))
        except (KeyboardInterrupt, EOFError):
            print("A: Goodbye")
            sys.exit()


if __name__ == "__main__":
    answer_loop()
