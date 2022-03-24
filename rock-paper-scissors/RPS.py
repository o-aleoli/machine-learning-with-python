# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

def Predict_Player(play_history, num):
    import pandas as pd
    import tensorflow as tf
    import os

    bots = ['mrugesh', 'abbey', 'quincy', 'kris']

    # Opponent history is a categorical feature, here encoded to a list of ints
    play_encoded = pd.DataFrame([0 if e == 'R' else (1 if e == 'P' else 2) for e in play_history], columns = ['plays'])
    play_encoded = play_encoded['plays'].values.reshape((1, play_encoded.shape[0]))
    play_encoded = tf.data.Dataset.from_tensor_slices(play_encoded).batch(1, drop_remainder = True)

    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                stateful = False
            ),
            batch_input_shape = (None, num, 1)
        ),
        tf.keras.layers.Dense(4, activation = 'softmax')# 'mrugesh', 'abbey', 'quincy' ou 'kris'
    ])
    # Network previouosly trained due to Replit's constrains for free accounts
    # Training carried with 400 plays, prediction done with 200 plays
    model.load_weights(os.path.join(os.getcwd(), 'trained_model.h5'))
    model.compile(
            loss = 'categorical_crossentropy',
            metrics = 'categorical_accuracy'
        )
    
    prediction = model.predict(play_encoded)

    return bots[prediction.argmax()]

def Choose_Strategy(opponent, player_prev_play, player_history, opponent_history, num_plays):
    if opponent == 'abbey':
        return Counter_Abbey(opponent_history, player_history, num_plays)
    elif opponent == 'kris':
        return Counter_Kris(player_prev_play)
    elif opponent == 'quincy':
        return Counter_Quincy([len(player_history)])
    else:
        return Counter_Mrugesh(player_history)

def Counter_Abbey(opponent_history=[],
            own_history=[],
            num_plays=0,
            play_order=[{
                "RR": 0,
                "RP": 0,
                "RS": 0,
                "PR": 0,
                "PP": 0,
                "PS": 0,
                "SR": 0,
                "SP": 0,
                "SS": 0,
            }]):

    if len(own_history) == num_plays:
        # Build statistics at first call to correctly apply abbey rules
        for i in range(1, len(own_history) - 1):
            play_pairs = ''.join(own_history[i - 1:i + 1])
            play_order[0][play_pairs] += 1

    # Proceed to play on abbey rules observing own plays
    ideal_response = {'P': 'R', 'R': 'S', 'S': 'P'}
    last_played = own_history[-1]
    last_two = ''.join(own_history[-2:])
    play_order[0][last_two] += 1

    potential_plays = [
        last_played + "R",
        last_played + "P",
        last_played + "S",
    ]

    sub_order = {k: play_order[0][k] for k in potential_plays if k in play_order[0]}

    prediction = max(sub_order, key=sub_order.get)[-1:]

    return ideal_response[prediction]

def Counter_Kris(prev_opponent_play):
    # Pick the play which counter kris's pick on its ideal_response
    ideal_response = {'P': 'R', 'R': 'S', 'S': 'P'}

    # Observe own's last play
    return ideal_response[prev_opponent_play]

def Counter_Quincy(counter=[0]):
    # Count the same way quincy counts
    # and pick from a table which wins
    # every play at the quincy's table
    choices = ['P', 'P', 'S', 'S', 'R']

    counter[0] += 1
    
    return choices[counter[0] % len(choices)]

def Counter_Mrugesh(opponent_history=[]):
    # Pick the play which counter mrugesh's pick on its ideal_response
    ideal_response = {'P': 'R', 'R': 'S', 'S': 'P'}
    
    # Measure own history's statistics
    last_ten = opponent_history[-10:]
    most_frequent = max(set(last_ten), key=last_ten.count)
    
    return ideal_response[most_frequent]

def player(opponent_prev_play,
        opponent_history=[],
        player_history=[],
        play_order=[{# Used against abbey
            "RR": 0,
            "RP": 0,
            "RS": 0,
            "PR": 0,
            "PP": 0,
            "PS": 0,
            "SR": 0,
            "SP": 0,
            "SS": 0,
        }]
    ):
    import random
    global prediction
    
    if not opponent_prev_play:
        # Reset at new opponent
        del opponent_history[:]
        del player_history[:]
        del play_order[:]
        play_order.append({
                "RR": 0,
                "RP": 0,
                "RS": 0,
                "PR": 0,
                "PP": 0,
                "PS": 0,
                "SR": 0,
                "SP": 0,
                "SS": 0,
        })
    else:
        opponent_history.append(opponent_prev_play)
    
    num_plays = 200
    counter = len(player_history)

    if counter < num_plays:
        # Play randomly before having enough history to use the model
        play = random.choice(['R', 'P', 'S'])
        player_history.append(play)
        return play
    
    elif counter == num_plays:
        # Use the model to identify the opponent and adopt the specific strategy
        # Model obtained from a bidirectional LSTM RNN trained from random plays
        # Details at ./rnn_train_module.py
        prediction = Predict_Player(opponent_history, num_plays)
        play = Choose_Strategy(prediction, player_history[-1], player_history, opponent_history, num_plays)
        player_history.append(play)
        return play
    
    else:
        # Continue to use the same strategy until the end
        play = Choose_Strategy(prediction, player_history[-1], player_history, opponent_history, num_plays)
        player_history.append(play)
        return play