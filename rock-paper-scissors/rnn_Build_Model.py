def Build_Model (seq_length, rnn_units, directory):
    import tensorflow as tf
    import tensorflow_addons as tfa

    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                rnn_units,
                stateful = False
            ),
            batch_input_shape = (None, seq_length, 1)
        ),
        tf.keras.layers.Dense(4, activation = 'softmax')# 'mrugesh', 'abbey', 'quincy' ou 'kris'
    ])
    model.compile(
        loss = 'categorical_crossentropy',
        metrics = tfa.metrics.F1Score(num_classes = 4, average = 'macro')
    )
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = directory,
        save_weights_only = True
    )

    return model, [callback]