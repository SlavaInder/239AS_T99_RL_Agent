import unittest
import os
import sys
import time
from copy import deepcopy

# list directories where packages are stored
# note that the parent directory of te repo is added automatically
GYM_FOLDER = "gym-t99"

# get this notebook's current working directory
nb_cwd = os.getcwd()
# get name of its parent directory
nb_parent = os.path.dirname(nb_cwd)
# add packages to path
sys.path.insert(len(sys.path), nb_parent)
sys.path.insert(len(sys.path), os.path.join(nb_parent, GYM_FOLDER))

import gym
registered = gym.envs.registration.registry.env_specs.copy()

from time import sleep
import gym_t99
from gym_t99.envs.state import Piece
import t_net
import numpy as np

class TestSuite(unittest.TestCase):
              
    # def setUp(self):
    #     pass
    
    # def tearDown(self):
    #     pass
    
    def test_observe_1(self):
        custom_gym = gym.make('gym_t99:t99sc-v0', num_players = 1, enemy="rand")

        action = {
        "reward": 0,
        "state": deepcopy(custom_gym.state)
        }

        custom_gym.step(action)

        observation = custom_gym._observe(0)
        total_sum = 0
        current_board = custom_gym.state.players[0].board

        
        for current_piece in [custom_gym.state.players[0].piece_current, custom_gym.state.players[0].piece_swap]:
            
            
            for i in range(4):
                temp_board = deepcopy(current_board)
                custom_gym._rotate_piece(current_board,current_piece)
                temp_board = custom_gym._apply_piece(temp_board,current_piece)

                #Find first nonzero
                left , right = temp_board.shape[1] , 0
                print(temp_board)
                for i in range(5):
                    #Find furthest left and right indices
                    row = temp_board[i, 3:-3]
                    indices = np.nonzero(row)[0]
                    length_ = len(indices)
                    if length_ == 0:
                        continue
                    left = min(left, indices[0])
                    right = max(right, indices[length_-1])
                print(left,right)
                width = (right - left) + 1
                zeros = 10 - width
                moves = zeros + 1
                total_sum += moves

        self.assertEqual(total_sum,len(observation[0]))
    
    def test_check_kos_1(self):
        custom_gym = gym.make('gym_t99:t99sc-v0', num_players = 1, enemy="rand")
        #Completely fill player's board
        board = custom_gym.state.players[0].board
    
        custom_gym._check_kos()
        self.assertEqual(custom_gym.active_players[0],True)

        board[3,6] = 1
        custom_gym._check_kos()
        self.assertEqual(custom_gym.active_players[0],False)

    def test_apply_garbage_1(self):
        custom_gym = gym.make('gym_t99:t99sc-v0', num_players = 1, enemy="rand")
        player = custom_gym.state.players[0]
        board = player.board
        player.incoming_garbage = [1,1,1,3,4,9]
        board[board.shape[0]-4,8] = 1
        board[board.shape[0]-4,6] = 5

        #print(board)
        custom_gym._apply_garbage(player)

        #print(board)

        desired_final_garbage = [2,3,8]
        self.assertEqual(player.incoming_garbage,desired_final_garbage)
        self.assertEqual(board[board.shape[0]-7,8] , 1)
        self.assertEqual(board[board.shape[0]-7,6] , 5)

    def test_clear_rows(self):
        custom_gym = gym.make('gym_t99:t99sc-v0', num_players = 1, enemy="rand")

        

        player = custom_gym.state.players[0]
        board = player.board

        #Fill bottom 2 rows
        height = board.shape[0]
        for j in range(board.shape[1]):
            if j > 2 and j < 13:
                board[height-4, j] = 4
        for j in range(board.shape[1]):
            if j > 2 and j < 13:
                board[height-5, j] = 5
        board, attack = custom_gym._clear_rows(board)

        #Verify rows were cleared
        for j in range(board.shape[1]):
            if j > 2 and j < 13:
                self.assertEqual(board[height-4, j],0)
        for j in range(board.shape[1]):
            if j > 2 and j < 13:
                self.assertEqual(board[height-5, j],0)
        self.assertEqual(attack, 2)

    

if __name__ == '__main__':
    unittest.main()
