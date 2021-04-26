import unittest
import os
import sys

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
import t_net


class TestSuite(unittest.TestCase):
              
    def setUp(self):
        self.custom_gym = gym.make('gym_t99:t99-v0', num_players = 2, enemy="rand")
    
    def tearDown(self):
        pass
    
    # def test_basic_debug_output(self):
    #     frame = self.custom_gym.render(mode="debug")
    #     print(frame[0])

    def test_window_show(self):
        #Window should show
        self.custom_gym.render(mode='human',show_window=True)
        sleep(4)
        self.custom_gym.close()
        sleep(4)
        
    # def test_window_no_show(self):
    #     #Window should not show
    #     self.custom_gym.render(mode='human',show_window=False)
    #     sleep(4)
    #     self.custom_gym.close()
    
    # def test_window_open_and_close(self):
    #     #Window should show
    #     self.custom_gym.render(mode='human',show_window=True)
    #     sleep(2)
    #     #Window should close, should show debug mode
    #     frame = self.custom_gym.render(mode='debug')
    #     sleep(4)
    #     self.custom_gym.render(mode='human',show_window=True)
    #     #Window should show again
    #     sleep(2)
    #     self.custom_gym.close()
    #     #Does not seem to work

    #To add pieces:
    #temp_board = self._apply_piece(self.state.players[i].board.copy(), self.state.players[i].piece_current)
    


if __name__ == '__main__':
    unittest.main()
