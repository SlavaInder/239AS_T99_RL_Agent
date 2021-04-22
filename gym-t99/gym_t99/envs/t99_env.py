import gym
from gym import error, spaces, utils
from gym.utils import seeding

class T99(gym.Env):
    metadata = {'render.modes': ['human', 'debug']}


    def __init__(self, enemy, num_players=2):
        """
        function to initialize gym environment. Action space is provided for reference
            0   -   do nothing
            1   -   choose random attack strategy
            2   -   choose to attack one that is closed to KO
            3   -   choose to attack one that has the highest KOs number
            4   -   choose to attack everybody who attacks you
            5   -   swap a piece
            6   -   move left
            7   -   move right
            8   -   turn clockwise 90 degrees
            9   -   turn counter clockwise 90 degrees

        :param enemy: a strategy able to produce actions based on observation; note, that this is not the strategy
                                    we are training, instead, this is a strategy we are training against. Usually,
                                    this is the previous iteration of AI
        :param num_players: number of competing agents (ideally, 99) in the game
        """
        self.action_space = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.enemy = enemy
        self.state = State99(num_players)


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
        self.state = State99()


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
