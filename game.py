# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:29:30 2016

@author: weinfz
"""
from flask import Flask
from othello1 import *
from flask_restful import reqparse, abort, Api, Resource
import json

app = Flask(__name__)
api = Api(app)

class Play(Resource):
    def get(self, player):
        if player == -1:
            player1 = BestPlayer(1)
            player1.load_model_from_pickel('rf1.pkl')
            player2 = HumanPlayer(-1)
        if player == 1:
            player1 = HumanPlayer(1)
            player2 = BestPlayer(-1)
            player2.load_model_from_pickel('rf1.pkl')
        self.game = Reversi(player1, player2)
        boards = self.game.get_moves_and_board(self.game.possession_arrow)
        return boards, self.board
        
    def post(self,player):
        
        

api.add_resource(Play, '/new_game/<int:player>')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)

