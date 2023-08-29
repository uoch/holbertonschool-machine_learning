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
        saver = tf.train.Saver()
        for i in range(epochs):
            count = 1
            cost, train_acc = sess.run((loss, accuracy), feed_dict={
                                       x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run(
                (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))
            X_sh, Y_sh = shuffle_data(X_train, Y_train)
            for j in range(0, len(X_train), batch_size):
                X_bat = X_sh[j:j+batch_size, ]
                Y_bat = Y_sh[j:j+batch_size, ]
                _ = sess.run(train_op, feed_dict={x: X_bat, y: Y_bat})
                if count % 100 == 0:
                    batch_cost, batch_acc = sess.run(
                        (loss, accuracy), feed_dict={x: X_bat, y: Y_bat})
                    print("\tStep {}:".format(count))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_acc))
                count += 1
        cost_f, train_acc_f = sess.run((loss, accuracy), feed_dict={
                                       x: X_train, y: Y_train})
        valid_cost_f, valid_acc_f = sess.run(
            (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(cost_f))
        print("\tTraining Accuracy: {}".format(train_acc_f))
        print("\tValidation Cost: {}".format(valid_cost_f))
        print("\tValidation Accuracy: {}".format(valid_acc_f))
        save_path = saver.save(sess, save_path)
    return save_path
