#!/usr/bin/python3

def main():
    from rnn_Build_Model import Build_Model
    import tensorflow as tf
    import pandas as pd
    import os

    path = os.getcwd()

    db_test = pd.read_csv(os.path.join(path, 'db_test.csv')).drop(columns = 'Unnamed: 0')
    labels_test = pd.get_dummies(db_test['bot'])
    features_test = db_test.drop(columns = 'bot')

    batch_size = 100

    TFd_plays_test = tf.data.Dataset.from_tensor_slices((
        features_test,
        labels_test
    ))\
        .batch(batch_size, drop_remainder = True)

    sequence_length = 400
    hidden_lstm_units = 2 * 64

    model, _ = Build_Model(sequence_length, hidden_lstm_units, path)
    model.load_weights(os.path.join(path, 'trained_model.h5'))
    model.evaluate(TFd_plays_test)

if __name__ == '__main__':
    main()