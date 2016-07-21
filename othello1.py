# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 07:40:22 2016

@author: weinfz
"""

import random
import sys
import numpy as np
from multiprocessing import Pool
import random
import pandas as pd
import copy
from sklearn.externals import joblib 
  
def weighted_random_choice(choices,weights):
    weights = [x/sum(weights) for x in weights] 
    cs = np.cumsum(weights) 
    idx = sum(cs < random.random()) 
    return choices[idx]
    
def _isValid(board,player,move):
    #print(move)
    if board[tuple(move)] != 0:
        return False
    to_flip = [move]
    directions = [(1,-1),
                    (1,0),
                    (1,1),
                    (0,-1),
                    (0,1),
                    (-1,-1),
                    (-1,0),
                    (-1,1)]
    for direction in directions:
        direction_moves = []
        for i in range(1,8):
            #print(direction,i)
            #import pdb; pdb.set_trace()
            check = tuple(np.array(move) + i*np.array(direction))
#            print(board)
#            print(tuple(check))
#            print(board[tuple(check)])
            if check[0] < 0 or check[1] < 0 or check[0] > 7 or check[1] > 7:
                #print('break')
                break
            
            elif board[check] == 0:
                #print('break')
                direction_moves = []
                break
                    
            elif board[check] == player:
                if i == 1:
                    #print('break')
                    #print(i)
                    break
                elif i>1:
                    #print(direction_moves,'X')
                    #print(i)
                    to_flip.extend(direction_moves)
                    break
                    
            elif board[check] == -1*player:
                direction_moves.append(check)
            else:
                print('hmm')
                
    if len(to_flip) == 1:
        return []
    return to_flip

class Player(object):
    def __init__(self, player):
        self.player = player
    
    def load_model(self, model):
        self.model = model
        
    def load_model_from_pickel(self, file):
        self.model = joblib.load(file)
        
    def _make_pred(self, state):
        _X = state.flatten()
        pred = self.model.predict_proba(_X)
        if pred.shape[1] == 3:
            pred = np.delete(pred,1,1)
        if self.player == 1:
            return pred[0][1]
        elif self.player == -1:
            return pred[0][0]
            
    def _make_preds(self,states,board=None, moves=None):
        probs = []
        for state in states:
            probs.append(self._make_pred(state))
        self.probs = probs
        
    def _choose_highest(self):
        try:
            return self.probs.index(max(self.probs))
        except Exception:
            import pdb; pdb.set_trace()
            
    def make_good_move(self, states,board=None, moves=None):
        self._make_preds(states)
        #import pdb; pdb.set_trace()
        return weighted_random_choice(states,self.probs)
        
    def make_move(self,states,board=None, moves=None):
        self._make_preds(states)
        #import pdb; pdb.set_trace()
        return states[self.probs.index(max(self.probs))]
        
class BestPlayer(Player):
    def make_good_move(self, states,board=None, moves=None):
        self._make_preds(states)
        #import pdb; pdb.set_trace()
        return states[self.probs.index(max(self.probs))]
  
             
class RandomPlayer(Player):
    def make_move(self,states,board=None, moves=None):
        return random.randint(0,(len(states)-1))
    
    def make_good_move(self,states,board=None, moves=None):
        choice = self.make_move(states)
        return choice
                    
class HumanPlayer(Player):
    def make_move(self, states,board=None, moves=None):
        print(board)
        print(moves)
        print(states)
        print(len(states)-1)
        choice = input("choose index number of choice: ")
        chosen_state = np.copy(states[int(choice)])
        #import pdb; pdb.set_trace()
        return chosen_state
    
    def make_good_move(self, states,board=None, moves=None):
        choice = self.make_move(states,board, moves)
        return choice


def _findMoves(board,player):
    moves = np.transpose(np.where(board == 0))
    #import pdb; pdb.set_trace()
    if len(moves) == 0:
        return False
    valid_moves = []
    for move in moves:
        move = tuple(move)
        valid_move = _isValid(board,player,move)
        if len(valid_move) != 0:
            #print(tuple(move))
            valid_moves.append(tuple(valid_move))
    return valid_moves
        
def _get_boards(board, moves, player):
    
    if moves == None:
        return None
    boards = []
    for move in moves:
        new_board = _apply_move(np.copy(board), move, player)
        boards.append(new_board)
    return boards
      
def _apply_move(board, move, player):
    for m in move:
        board[m] = player
    return board

class Reversi(object):
    def __init__(self,positive,negative):
        """
        black is 1 white is -1
        """
        self.board = np.zeros((8,8),dtype=np.int8)
        self.board[4,3:5] = [1,-1]
        self.board[3,3:5] = [-1,1]
        self.possession_arrow = 1
        self.game_over = False
        self.boards = [self.board.flatten()]
        self.player = {-1:negative,
                       1:positive}
        
    def _check_poss(self, player):
        if self.possession_arrow == player:
            pass
        else:
            Exception('its not hos turn')
            
    def _get_moves(self, player):
        moves = _findMoves(self.board, player)
        if moves == False:
            self._game_over()
            return 
        if len(moves) == 0:
            other_moves = _findMoves(self.board, player*-1)
            if len(other_moves) == 0:
                self._game_over()
                return 
            return
        return moves
        
    def get_moves_and_board(self,player):
        if self.game_over == True:
            #print('game already ended')
            return 
        moves = self._get_moves(player)
        boards = _get_boards(self.board, moves, player)
        if boards == None:
            #print('game already_over')
            return
        boards_dict = {str(moves[i][0]) : boards[i].tolist() for i in range(len(moves))}
        return boards_dict
        
                
    def make_move(self,player):
        #print(self.game_over)
        #import pdb; pdb.set_trace()
        if self.game_over == True:
            #print('game already ended')
            return 
        moves = self._get_moves(player)
        boards = _get_boards(self.board, moves, player)
        if boards == None:
            #print('game already_over')
            return
        
        picked_board = self.player[player].make_good_move(boards,self.board,moves)
        #import pdb; pdb.set_trace()
        #picked_board = boards[board_index]
        #import pdb; pdb.set_trace()
        self.boards.append(picked_board.flatten())
        self.board = picked_board

        
        
    def _game_over(self):

        #print('game over')
        players, new_board = np.unique(self.board.flatten(),return_inverse=True)
        score = np.bincount(new_board)
        results = {players[i]:score[i] for i in range(len(players))}
        self.game_over = True
        self.results = {'score':results,
                'states':self.boards}
                
    def play_game(self):
        for i in range(100):
            #import pdb; pdb.set_trace()
            self.make_move(self.possession_arrow)
            self.possession_arrow *= -1
            #import pdb; pdb.set_trace()
            #print(self.game_over)
            if self.game_over == True:
                return self.results
            if i>64:
                raise Exception('over max turns')

def addData(number_of_games,file):
    player1 = Player(1)
    player1.load_model_from_pickel('rf1.pkl')
    player2 = Player(-1)
    player2.load_model_from_pickel('rf1.pkl')
    headers = []
    for x in range(8):
        for y in range(8):
            headers.append(str(x)+str(y))
    for i in range(number_of_games):
        if random.random() > .5:
            game = Reversi(player1, player2)
        else:
            game = Reversi(player2, player1)
        results = game.play_game()
        table = pd.DataFrame(results['states'],columns=headers)
        table['first'] = results['score'][1] 
        table['second'] = results['score'][-1] 
#        win_loss = {-1:results['score'][-1],
#                    1:results['score'][1]}   
        win_loss = table.loc[0,['first','second']]
        win_loss['first_win'] = win_loss['first'] > win_loss['second']
        win_loss = win_loss.to_frame().T
        with open('data1.csv', 'a') as f:
            
            table.to_csv(f,index=False,header=False)
        with open ('win_loss1.csv', 'a') as f:
            win_loss.to_csv(f,index=False, header=False)


def sim(x):      
    file = 'data1.csv'
    addData(x,file)

def PlayRF():
    headers = []
    for x in range(8):
        for y in range(8):
            headers.append(str(x)+str(y))
    player1 = BestPlayer(1)
    player1.load_model_from_pickel('rf.pkl')
    player2 = HumanPlayer(-1)
    if random.random() > .5:
        game = Reversi(player1, player2)
    else:
        game = Reversi(player2, player1)
    results = game.play_game()
    print(results)
    table = pd.DataFrame(results['states'],columns=headers)
    table['first'] = results['score'][1] 
    table['second'] = results['score'][-1] 
    win_loss = table.loc[0,['first','second']]
    win_loss['first_win'] = win_loss['first'] > win_loss['second']
    win_loss = win_loss.to_frame().T
    with open('data1.csv', 'a') as f:
        
        table.to_csv(f,index=False,header=False)
    with open ('win_loss1.csv', 'a') as f:
        win_loss.to_csv(f,index=False, header=False)
    
    
    

def PlayPara(number_workers,number_series):
    pool = Pool()
    numbers_of_sims = [number_series for x in range(number_workers)]
    result_final = pool.map(sim, numbers_of_sims)
    pool.close()

if __name__ == "__main__":
    PlayRF()
    #sim(1000)
    #PlayPara(4,100)





