#!/usr/bin/python3

def main():
    from rnn_Build_Model import Build_Model
    import tensorflow as tf
    import os
    import pandas as pd

    sequence_length = 400
    hidden_lstm_units = 2 * 64
    batch_size = 100
    number_epochs = 3
    path = os.getcwd()

    # Generated with ./generate_ETL_plays.py
    # 45k shuffled sequences of 400 random plays
    db_train = pd.read_csv(os.path.join(path, 'db_train.csv'))\
        .drop(columns = 'Unnamed: 0')
    
    labels_train = pd.get_dummies(db_train['bot'])
    features_train = db_train.drop(columns = 'bot')

    TFd_plays_train = tf.data.Dataset.from_tensor_slices((
        features_train,
        labels_train
    ))\
        .batch(batch_size, drop_remainder = True)

    path_ckpt = os.path.join(os.getcwd(), 'training_ckpts', 'ckpt_{epoch}')
    model, ckpt_callback = Build_Model(sequence_length, hidden_lstm_units, path_ckpt)

    continuation_job = False
    if continuation_job:
        model.load_weights(tf.train.latest_checkpoint(path_ckpt))

    model.fit(
        TFd_plays_train,
        epochs = number_epochs,
        callbacks = ckpt_callback
    )
    model.save_weights(os.path.join(path, 'trained_model.h5'))

if __name__ == '__main__':
    main()