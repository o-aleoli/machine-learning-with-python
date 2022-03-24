#!/usr/bin/python3

def main():
    '''
    Validation strategy differs from train and test procedures.
    Here, the model is evaluated at RPS' conditions,
    with reduced play sequence to maximize the win ratio.
    '''
    from rnn_Build_Model import Build_Model
    import os
    import tensorflow as tf
    import pandas as pd

    path = os.getcwd()
    bots = ['mrugesh', 'abbey', 'quincy', 'kris']

    # Generated with ./generate_ETL_plays_validation.py
    db_validation = pd.read_csv(os.path.join(path, 'db_validation.csv')).drop(columns = 'Unnamed: 0')
    labels_validation = pd.get_dummies(db_validation['bot'])
    features_validation = db_validation.drop(columns = 'bot')

    TFd_plays_validation = tf.data.Dataset.from_tensor_slices((
        features_validation,
        labels_validation
    ))\
        .batch(1, drop_remainder = True)

    sequence_length = 200
    hidden_lstm_units = 2 * 64

    model, _ = Build_Model(sequence_length, hidden_lstm_units, path)
    model.load_weights(os.path.join(path, 'trained_model.h5'))
    prediction = model.predict(TFd_plays_validation)

    for i in range(len(bots)):
        print(f'{bots[prediction[i].argmax()]: <7} ({bots[labels_validation.to_numpy()[i].argmax()]})')

if __name__ == '__main__':
    main()