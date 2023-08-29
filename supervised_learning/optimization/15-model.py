#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy"""
    x = tf.cast(tf.shape(y)[0], tf.float32)
    y_pred = tf.argmax(y_pred, axis=1)
    y = tf.argmax(y, axis=1)
    true_false_array = tf.equal(y_pred, y)
    num_of_true = tf.cast(tf.reduce_sum(
        tf.cast(true_false_array, tf.int32)), tf.float32)
    accuracy = num_of_true / x
    mean_accuracy = tf.reduce_mean(accuracy, name="Mean")
    return mean_accuracy


def calculate_loss(y, y_pred):
    """Calculates the loss for a given y_pred"""
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred,)
    return loss


def create_batch_norm_layer(prev, n, activation):
    """
    apply the activation function to the normalized inputs
    """
    kernal = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n, activation=None,
        kernel_initializer=kernal)
    z = layer(prev)
    if activation == None:
        return z
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8
    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, epsilon)
    return activation(z_norm)


def forward_prop(prev, layers, activations, epsilon):
    """all layers get batch_normalization but the last one, that stays 
    without any activation or normalization"""
    for i in range(len(layers)):
        prev = create_batch_norm_layer(prev, layers[i], activations[i])
    return prev


def shuffle_data(X, Y):
    # fill the function
    ind = np.random.permutation(len(X))
    return X[ind], Y[ind]


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learning rate decay operation in tensorflow using inverse time decay"""
    le = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    return le


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    epsilon is a small number to avoid division by zero
    """
    op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return op.minimize(loss)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('y', y)
    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)
    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)
    tf.add_to_collection('global_step', global_step)
    # compute decay_steps
    decay_step = X_train.shape[0] // batch_size
    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_step)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            # print training and validation cost and accuracy
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
            # shuffle data
            X_sh, Y_sh = shuffle_data(X_train, Y_train)
            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                X_bat = X_sh[j:j+batch_size, :]  # conserve all the columns
                Y_bat = Y_sh[j:j+batch_size, :]  # conserve all the columns
                sess.run(train_op, feed_dict={
                         x: X_bat, y: Y_bat, global_step: i})
                if count % 100 == 0:
                    batch_cost, batch_acc = sess.run(
                        (loss, accuracy), feed_dict={x: X_bat, y: Y_bat})
                    print("\tStep {}:".format(count))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_acc))
                count += 1
                # run training operation
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
    # print batch cost and accuracy

    # print training and validation cost and accuracy again

    # save and return the path to where the model was saved
