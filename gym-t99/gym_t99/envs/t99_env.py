import gym
from gym import error, spaces, utils
from gym.utils import seeding

class T99(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self):
        # here we should download saved weights of previous player
        #
        print('init t99 2')


    def step(self, action):
        """
        the function which makes one update of the environment. Ends with fixing    

        :param dict action:
        :return object observation: full state of the game board.
        :return float observation: amount of reward achieved by the previous action
        :return boolean done: whether it’s time to reset the environment again, True if the agent loses or wins
        :return dict info: diagnostic information useful for debugging.
        """

    def reset(self):
        print('reset')


    def render(self, mode='human'):
        print('render')


    def close(self):
        print('close')
