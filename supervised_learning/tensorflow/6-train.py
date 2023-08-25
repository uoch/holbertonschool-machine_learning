#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """prepar graph then run it in a session"""
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)
    # can you give me how i could use tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Initialize variables
        for i in range(iterations + 1):
            sess.run(init)
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(
                    sess.run(loss, feed_dict={x: X_train, y: Y_train})))
                print("\tTraining Accuracy: {}".format(
                    sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
                print("\tValidation Cost: {}".format(
                    sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
                print("\tValidation Accuracy: {}".format(
                    sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
                if i < iterations:
                    sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)

        return save_path
