#!/usr/bin/python3

def Play_List(player1, player2, num_games):
    # Simplified play function from RPS_game.py
    p1_prev_play = ""
    p2_prev_play = ""
    bot_history = []
    
    for _ in range(num_games):
        p1_play = player1(p2_prev_play)
        p2_play = player2(p1_prev_play)

        bot_history.append(p2_play)

        p1_prev_play = p1_play
        p2_prev_play = p2_play
    
    return bot_history

def main():
    from RPS_game import mrugesh, abbey, quincy, kris, random_player
    import pandas as pd
    import os
    import numpy as np

    num_games = int(5e6)
    rand_mrugesh = Play_List(random_player, mrugesh, num_games)
    rand_abbey = Play_List(random_player, abbey, num_games)
    rand_quincy = Play_List(random_player, quincy, num_games)
    rand_kris = Play_List(random_player, kris, num_games)

    out = [
        pd.DataFrame(rand_mrugesh, columns = ['play']),
        pd.DataFrame(rand_abbey, columns = ['play']),
        pd.DataFrame(rand_quincy, columns = ['play']),
        pd.DataFrame(rand_kris, columns = ['play'])
    ]
    plays = ['R', 'P', 'S']
    bots = ['mrugesh', 'abbey', 'quincy', 'kris']

    for i in range(len(bots)):
        out[i] = out[i].applymap(plays.index)
        out[i]['bot'] = i

    db_plays = pd.concat(out, axis = 0).reset_index(drop = True)
    
    sequence_length = 400

    # Transformation from a tidy table to a long table
    examples = pd.DataFrame(db_plays['play'].values.reshape((db_plays.shape[0] // sequence_length, sequence_length)))
    labels = db_plays.loc[::sequence_length, 'bot']

    for i in range(examples.shape[0]):
        examples.loc[i, 'bot'] = labels.iloc[i]

    examples = examples.reindex(np.random.permutation(examples.index))

    partitions = [
        int(0.9 * examples.shape[0]),# 90% (45k) for training
        examples.shape[0]# 10% (5k) for testing
    ]
    db_filenames = ['db_train.csv', 'db_test.csv']
    dbs = [
        examples.iloc[:partitions[0]],
        examples.iloc[partitions[0]:partitions[1]]
    ]

    for i in range(len(db_filenames)):
        dbs[i].to_csv(os.path.join(os.getcwd(), db_filenames[i]))

if __name__ == '__main__':
    main()