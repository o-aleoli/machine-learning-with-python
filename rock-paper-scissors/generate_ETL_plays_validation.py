#!/usr/bin/python3

def main():
    from RPS_game import mrugesh, abbey, quincy, kris, random_player
    from generate_ETL_plays import Play_List
    import pandas as pd
    import numpy as np
    import os

    num_games = int(399)

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
    bots = ['mrugesh', 'abbey', 'quincy', 'kris']
    plays = ['R', 'P', 'S']

    for i in range(len(bots)):
        out[i] = out[i].applymap(plays.index)
        out[i]['bot'] = i

    db_plays = pd.concat(out, axis = 0).reset_index(drop = True)
    
    examples = pd.DataFrame(db_plays['play'].values.reshape((4, num_games)))
    labels = db_plays.loc[::num_games, 'bot']

    for i in range(examples.shape[0]):
        examples.loc[i, 'bot'] = labels.iloc[i]

    examples.columns = examples.columns.astype('str')
    examples = examples.reindex(np.random.permutation(examples.index))
    examples.to_csv(os.path.join(os.getcwd(), 'db_validation.csv'))

if __name__ == '__main__':
    main()