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
        # counter of steps made
        self.current_step = 0
        # how many moves per environment update a player can do
        self.update_frequency = 3


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

        # TODO: disable players who already lost

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
                temp_board = self._apply_piece(self.state.players[i].board.copy(), self.state.players[i].piece_current)
                # append the list with their board
                frame.append(temp_board)

        elif mode=="human":
            # TODO: Ian's code here
            frame=None

        return frame


    def close(self):
        print('close')


    def _process_event(self, event):
        # function that processes the following events: player's attack, ???
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


    def _update_player(self, player_id):
        """
        function that waits drops player's piece by 1 if this drop is possible;
        if the drop is impossible, it first adds the current piece to the board; then it iteratively deletes lines that
        can be cleared and shifts all lines on the top to fill missing row; then the attack event is created depending
        on how the lines were cleared. After everything is up to date, we check if the player lost
        """
        # try to move piece to the bottom
        success = self._move(self.state.players[player_id].board,
                             self.state.players[player_id].piece_current,
                             0, 1)
        # if drop is impossible, start update procedure;
        if not success:
            # calculate board's width
            b_width = self.state.players[player_id].board.shape[1]
            # add piece to the board
            self.state.players[player_id].board = self._apply_piece(self.state.players[player_id].board,
                                                                    self.state.players[player_id].piece_current)
            # check which lines are cleared
            cleared = np.prod(self.state.players[player_id].board.astype(bool), axis=1)
            print(cleared)
            # save the number of lines cleared to calculate attack power
            attack = np.sum(cleared)
            # for each cleared line
            i = len(cleared) - 3
            while i > 4:
                # if the line needs to be cleared
                if cleared[i] > 0:
                    # clear the line
                    self.state.players[player_id].board[i, 2:b_width-2] = 0
                    i -= 1
                else:
                    i -= 1


            # update piece at hand
            self._next_piece(self.state.players[player_id])


    def _apply_piece(self, board, piece):
        # stick piece to the board, and return new board
        board[piece.y-2:piece.y+3, piece.x-2:piece.x+3] += piece.matrix
        return board

    def _collision(self, board, piece):
        # check whether at leat one element of the piece overlaps wit board
        collided = np.sum(board[piece.y-2:piece.y+3, piece.x-2:piece.x+3].astype(bool)*piece.matrix.astype(bool))
        if collided > 0:
            return True
        else:
            return False

    def _move(self, board, piece, dx, dy):
        # moves a piece if possible. returns True if successfull, False if the move is impossible
        # update coordinates
        piece.x += dx
        piece.y += dy
        # check if the elements collided
        if self._collision(board, piece):
            # if collided, return coordinates back and exit
            piece.x -= dx
            piece.y -= dy
            return False
        else:
            # if successfull, exit
            return True

    def _next_piece(self, player):
        # change current piece
        player.piece_current = player.piece_queue.pop(0)
        # produce a new piece for the queue
        player.piece_queue.append(Piece())