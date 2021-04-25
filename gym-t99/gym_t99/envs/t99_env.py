import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .state import *

class T99(gym.Env):
    """
    public API consists only of 4 funcs:
        - step
        - reset
        - render
        - close
    """
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
        reward = None
        next_state = None
        done = None
        info = {}

        # step 1: process all events
        for event in self.state.event_queue:
            self._process(event)
        # step 2: process all actions
        for i in range(len(self.state.players)):
            # if this is the first player, controlled by AI
            if i==0:
                # use action passed in command option of step
                self._apply_action(i, action)
            # if this player is controlled by environment AI
            else:
                # first, generate action
                action = self.enemy.action(self.state.observe(i))
                # and then apply it
                self._apply_action(i, action)
        # step 3: update all player's with in-game mechanics
        for i in range(len(self.state.players)):
            self._update_player(i)

        return next_state, reward, done, info


    def reset(self):
        self.state = State99()


    def render(self, mode='human'):
        """
        :param str mode: mode in which rendering works. If debug, returns numpy matrices for each player
        :return object frame: depending on mode, renders a human-visible screenshot of the game
        """
        if mode=="debug":
            # create a list for players' boards
            frame = []
            # for each player
            for i in range(len(self.state.players)):
                # copy the board together with piece
                temp_board = self._apply(self.state.players[i].board, self.state.players[i].piece_current)
                # append the list with their board
                frame.append(temp_board)

        elif debug=="human":
            # TODO:  Ian's code here
            frame=None

        return frame


    def close(self):
        print('close')


    def _apply(self, board, piece):
        # stick piece to the board, and return new board
        board[piece.y-2:piece.y+3, piece.x-2:piece.x+3] += piece.matrix
        return board

    def _collision(self, board, piece):
        # check whether at leat one element of the piece overlaps wit board
        collided = np.sum(board[piece.y-2:piece.y+3, piece.x-2:piece.x+3]+piece.matrix)
        if collided > 0:
            return True
        else:
            return False

    def _process_event(self, event):
        # function that processes the following events: player's attack, ???
        pass

    def _update_player(self, player_id):
        # function that drops player's piece by 1, clears lines if this drop is impossible,
        pass


    def _apply_action(self, player_id, action):
        """
        :param int num_player: the id of the player who made the action
        :param action: the id of the action we have to perform
        """
        # go through all options of action id-s and perform them
        if action == 1:
            # fill your logic here
            pass
        elif action == 2:
            pass
        # and so on and so forth

