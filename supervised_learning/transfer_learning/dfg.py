    import tensorflow as tf
    print(tf.__version__)
    input = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Dense(64, activation='relu')(input)
    y = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(y)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()