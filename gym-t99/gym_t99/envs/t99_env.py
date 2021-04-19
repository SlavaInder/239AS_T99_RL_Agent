import gym
from gym import error, spaces, utils
from gym.utils import seeding

class T99(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, num_players=2):
        # here we should download saved weights of previous player
        #

        # TODO: Discuss in meeting. 
        self.action_space = None
        self.observation_space = None
        self.state = None


        print('init t99 2')


    def step(self, action):
        """
        the function which makes one update of the environment. Ends with fixing    

        :param dict action:
        :return object observation: full state of the game board.
        :return float reward: amount of reward achieved by the previous action
        :return boolean done: whether it’s time to reset the environment again, True if the agent loses or wins
        :return dict info: diagnostic information useful for debugging.
        """
        reward = self.reward(action)
        next_state = self.next_state(action)
        done = self.is_done()
        info = {}
        return next_state, reward, done, info



    def reset(self):
        print('reset')


    def render(self, mode='human'):
        print('render')


    def close(self):
        print('close')


    def reward(self, action):
        """
        Helper to return reward based on action.
        """
        return -1.0

    def next_state(self, action):
        return None

    def is_done(self):
        """Helper to return if terminal state is reached.
        """
        if self.state:
            return False
        else:
            return True
