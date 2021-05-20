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

    """            Public Methods here             """

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
        self.num_players = num_players
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.enemy = enemy
        self.state = State99(num_players)
        # an array to keep track of who is in the game
        self.active_players = np.ones(num_players).astype(bool)
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
        done = False
        info = {}

        # TODO: disable players who already lost

        # step 1: process all events
        for event in self.state.event_queue:
            self._process_event(event)
        # step 2: process all actions
        for i in range(len(self.state.players)):
            # if this is the first player, controlled by AI, and it is still active
            if i == 0 and self.active_players[i]:
                # use action passed in command option of step
                self._apply_action(i, action)
            # if this player is controlled by environment AI, and it is still active
            elif self.active_players[i]:
                # first, generate action
                action = self.enemy.action(self.state.observe(i))
                # and then apply it
                self._apply_action(i, action)
        # step 3: update all player's with in-game mechanics
        for i in range(len(self.state.players)):
            # if the player is still active
            if self.active_players[i]:
                self._update_player(i)
        # if either the AI has lost or all its enemies lost
        if (not self.active_players[0]) or \
                (len(self.active_players) > 1 and np.prod(self.active_players[1:])):
            # then we are done for thi
            # s round
            done = True

        next_state = self._observed_state()
        reward = self.vanilla_tetris_reward()

        return next_state, reward, done, info

    def reset(self):
        self.current_step = 0
        self.state = State99(self.num_players)
        self.active_players = np.ones(self.num_players).astype(bool)
        return (self._observed_state(), 0, False, None)

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

    """            Main Loop Methods here             """

    def _process_event(self, event):
        # function that processes the following events: player's attack, ???
        pass

    def _apply_action(self, player_id, action):
        """
        :param int num_player: the id of the player who made the action
        :param action: the id of the action we have to perform
        """
        # go through all options of action id-s and perform them
        # choose attack strategy
        if action in [6, 7, 8, 9]:
            self.state.players[player_id].attack_strategy = action
        # swap a piece
        # note that after a piece was swapped, new piece starts at the top to avoid conflicts at collisions
        # this means that after a piece was placed, swapped piece corrdinates return to the top
        if action == 5:
            self.state.players[player_id].piece_current, self.state.players[player_id].piece_swap = \
                self.state.players[player_id].piece_swap, self.state.players[player_id].piece_current
        # Move piece left
        if action == 0:
            success = self._move(self.state.players[player_id].board,
                                 self.state.players[player_id].piece_current,
                                 -1, 0)
        # Move piece right
        elif action == 1:
            success = self._move(self.state.players[player_id].board,
                                 self.state.players[player_id].piece_current,
                                 1, 0)
        # Move piece clockwise 90 degrees
        elif action == 2:
            self._rotate_piece(self.state.players[player_id].board,
                               self.state.players[player_id].piece_current,
                               clockwise=True)

        # Move piece counter clockwise 90 degrees
        elif action == 3:
            self._rotate_piece(self.state.players[player_id].board,
                               self.state.players[player_id].piece_current,
                               clockwise=False)

    def _update_player(self, player_id):
        """
        function that waits drops player's piece by 1 if this drop is possible;
        if the drop is impossible, it first adds the current piece to the board; then it iteratively deletes lines that
        can be cleared and shifts all lines on the top to fill missing row; then the attack event is created depending
        on how the lines were cleared. After everything is up to date, we check if the player lost
        """
        # calculate board's width
        b_height, b_width = self.state.players[player_id].board.shape
        # try to move piece to the bottom
        success = self._move(self.state.players[player_id].board,
                             self.state.players[player_id].piece_current,
                             0, 1)
        # if drop is impossible, start update procedure;
        if not success:
            # add piece to the board
            self.state.players[player_id].board = self._apply_piece(self.state.players[player_id].board,
                                                                    self.state.players[player_id].piece_current)
            # check which lines are cleared
            cleared = np.prod(self.state.players[player_id].board.astype(bool), axis=1)
            # save the number of lines cleared to calculate attack power
            attack = np.sum(cleared)
            self.state.players[player_id].num_lines_recently_cleared = attack
            # for each cleared line
            i = len(cleared) - 4
            while i > 4:
                # if the line needs to be cleared
                if cleared[i] > 0:
                    # clear the line
                    self.state.players[player_id].board[i, 3:b_width-3] = 0
                    cleared[i] = 0
                    # shift all lines from the top by 1
                    self.state.players[player_id].board[6:i+1, 3:b_width-3] = \
                        self.state.players[player_id].board[5:i, 3:b_width-3]
                    cleared[6:i+1] = cleared[5:i]
                    # clear the top line, which does not have pieces after shift
                    self.state.players[player_id].board[5, 3:b_width-3] = 0
                    cleared[5] = 0
                else:
                    i -= 1
            # TODO: add attack event here

            # update piece at hand
            self._next_piece(self.state.players[player_id])
            # reset coordinates of the swap piece
            self.state.players[player_id].piece_swap.y = 3
            self.state.players[player_id].piece_swap.x = np.random.randint(5, high=b_width-5)

        # check if player lost
        if np.sum(self.state.players[player_id].board.astype(bool)[0:5, 3:b_width-3]) > 0:
            # assign the position in the leaderboard
            position = len(self.active_players) - np.sum(np.where(self.active_players is True, 1, 0))
            self.state.players[player_id].place = position
            # if so, update the list of active players
            self.active_players[player_id] = False

    """         Helper functions here           """

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

    def _rotate_piece(self, board, piece, clockwise=True):  
        # rotates a piece clockwise if possible
        if clockwise:
            piece.matrix = np.rot90(piece.matrix, axes=(1, 0))
        else:
            piece.matrix = np.rot90(piece.matrix, axes=(0, 1))
        # check if the elements collided
        if self._collision(board, piece):
            if clockwise:
                piece.matrix = np.rot90(piece.matrix, axes=(0, 1))
            else:
                piece.matrix = np.rot90(piece.matrix, axes=(1, 0))
            return False
        else:
            # if successfull, exit
            return True


    def _next_piece(self, player):
        # change current piece
        player.piece_current = player.piece_queue.pop(0)
        # produce a new piece for the queue
        player.piece_queue.append(Piece())


    def _observed_state(self):
        return_state = []
        for i, player in enumerate(self.state.players):
            # return everything related to the current player.
            if i == 0:
                return_state.append((self._apply_piece(self.state.players[i].board.copy(), self.state.players[i].piece_current),
                    self.state.players[i].board,
                    self.state.players[i].piece_current.roll,
                    self.state.players[i].piece_swap,
                    self.state.players[i].KOs,
                    self.state.players[i].incoming_garbage,
                    self.state.players[i].place,
                    self.state.players[i].attack_strategy,
                    self.state.players[i].num_lines_recently_cleared))
            # Otherwise return only the board and the number of badges (number of KOs in our case).
            else:
                return_state.append((self.state.players[i].board, self.state.players[i].KOs))

        return return_state

    def vanilla_tetris_reward(self):
        return 1 + ((self.state.players[0].num_lines_recently_cleared ** 2) * self.state.players[0].board.shape[1])