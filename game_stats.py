# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:07:12 2016

@author: weinfz
"""

import pandas as pd
import numpy as np
headers = ['first','second','first_win','winner','loser']
data = pd.read_csv('win_loss2.csv',header=None,names=headers)

players = np.unique(data[['winner', 'loser']].values).tolist()
stats = []
for player in players:
    wins = len(data.loc[data['winner'] == player].index)
    losses = len(data.loc[data['loser'] == player].index)
    pct = wins/(wins+losses)
    stats.append({'player':player, 'wins':wins, 'losses':losses, 'pct':pct})
    
stats = pd.DataFrame(stats)   

h2hstats = []
for player1 in players:
    for player2 in players: 
        same_game_indexes = (((data['winner'] == player1) & (data['loser'] == player2)) |
        ((data['winner'] == player2) & (data['loser'] == player1)))
        da = data.loc[same_game_indexes]
        wins = len(da.loc[da['winner'] == player1].index)
        losses = len(da.loc[da['loser'] == player1].index)
        try:
            pct = wins/(wins+losses)
        except ZeroDivisionError:
            pct = np.nan
        h2hstats.append({'player1':player1,'player2':player2, 'p1_wins':wins, 'p1_losses':losses, 'p1_pct':pct})
h2hstats = pd.DataFrame(h2hstats)



