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


