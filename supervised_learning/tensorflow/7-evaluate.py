#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """evaluate a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, "{}".format(save_path))
        # how to access to a variable that is not in the graph collection
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        # how to access to a op that is not in the graph collection
        y_pred = tf.get_collection("y_pred")[0]
        # how to access to a op that is not in the graph collection
        loss = tf.get_collection("loss")[0]
        # how to access to a op that is not in the graph collection
        accuracy = tf.get_collection("accuracy")[0]
        y_pred, accuracy, loss = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y})
        return y_pred, accuracy, loss
