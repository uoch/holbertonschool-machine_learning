#!/usr/bin/env python3
"""Normalization Constants"""""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """trains a load neural network model using mini-batch gradient descent"""
    with tf.Session() as sess:
        graph = tf.train.import_meta_graph(load_path+".meta")
        saver = tf.train.Saver()
        graph.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess.run(init)
        for i in range(epochs):
            count = 1
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(
                sess.run(loss, feed_dict={x: X_train, y: Y_train})))
            print("\tTraining Accuracy: {}".format(
                sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
            print("\tValidation Cost: {}".format(
                sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
            print("\tValidation Accuracy: {}".format(
                sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
            X_sh, Y_sh = shuffle_data(X_train, Y_train)
            for j in range(0, X_train.shape[0], batch_size):
                X_bat = X_sh[j:j+batch_size]
                Y_bat = Y_sh[j:j+batch_size]
                sess.run(train_op, feed_dict={x: X_bat, y: Y_bat})
                if count % 100 == 0:
                    print("\tStep {}:".format(count))
                    print("\t\tCost: {}".format(
                        sess.run(loss, feed_dict={x: X_bat, y: Y_bat})))
                    print("\t\tAccuracy: {}".format(
                        sess.run(accuracy, feed_dict={x: X_bat, y: Y_bat})))
                count += 1
        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(
            sess.run(loss, feed_dict={x: X_train, y: Y_train})))
        print("\tTraining Accuracy: {}".format(
            sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
        print("\tValidation Cost: {}".format(
            sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
        print("\tValidation Accuracy: {}".format(
            sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
        save_path = saver.save(sess, save_path)
    return save_path
