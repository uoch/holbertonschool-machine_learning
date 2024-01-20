#!/usr/bin/env python3
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """creates all masks for training/validation"""
    batch_size, seq_len_in = inputs.shape
    batch_size, seq_len_out = target.shape
    # padding mask for encoder
    