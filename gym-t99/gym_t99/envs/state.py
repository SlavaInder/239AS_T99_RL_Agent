import numpy as np


# these global variables should NEVER be accessed outside of this file to avoid name conflicts
# instead, use array.shape() to check for collisions
BOARD_WIDTH = 10
BOARD_HEIGHT = 20


class Piece:
    """
    class used to represent:
        - current piece of a player
        - swap piece of the player
        - pieces, that lie in the queue

    has 2 fields:
        - integer matrix 5x5, that represents form of the piece. These matrix can be rotated.
        - coordinate, that represents the position of the center of a piece, counting from top-left angle of the board
    """
    shapes = {
        # line piece
        1: np.array([[0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0]],
                    dtype=np.int16),

        # angle-shaped piece - right
        2: np.array([[0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0],
                     [0, 0, 2, 0, 0],
                     [0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.int16),

        # angle-shaped piece - left
        3: np.array([[0, 0, 0, 0, 0],
                     [0, 3, 3, 0, 0],
                     [0, 0, 3, 0, 0],
                     [0, 0, 3, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.int16),

        # t-shaped piece
        4: np.array([[0, 0, 0, 0, 0],
                     [0, 4, 4, 4, 0],
                     [0, 0, 4, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.int16),

        # stair piece - right
        5: np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 5, 0],
                     [0, 0, 5, 5, 0],
                     [0, 0, 5, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.int16),

        # stair piece - left
        6: np.array([[0, 0, 0, 0, 0],
                     [0, 6, 0, 0, 0],
                     [0, 6, 6, 0, 0],
                     [0, 0, 6, 0, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.int16),

        # square piece
        7: np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 7, 7, 0],
                     [0, 0, 7, 7, 0],
                     [0, 0, 0, 0, 0]],
                    dtype=np.int16),

    }

    # this function spawns a random piece
    def __init__(self):
        # roll a random number
        roll = np.random.choice(list(Piece.shapes.keys()))
        # copy a piece from template
        self.matrix = Piece.shapes[roll].copy()
        # sample a random x coordinate for a piece
        # make sure that it does not collide with the borders
        self.x = np.random.randint(3, high=BOARD_WIDTH-1)
        # set coordinate. It is constant for all new pieces
        self.y = 2

    def rotate_clockwise(self):
        self.matrix = np.rot90(self.matrix, axes=(1, 0))

    def rotate_counterclockwise(self):
        self.matrix = np.rot90(self.matrix, axes=(0, 1))


class Player99:
    """
    Class representing all features specific to a player.
    The instances of this class store:
        - current piece of a player
        - swap piece of the player
        - pieces, that lie in the queue
        - board, which is a matrix (BOARD_HEIGHT+4)x(BOARD_WIDTH+4), where 2 - size of the wall
        - KOs, the number of players kicked out by this AI. The higher is number of KOs, the
                    higher is attack multiplier
        - place, which is necessary to create leaderboard
        - incoming garbage, which is a list of ints numbers. Each int representing one line of garbage, that would
                    be sent to this AI after the number reaches zero. After each step, all numbers in the list are
                    decreased by 1. Then, all zeros are deleted from the list and garbage is added to the
                    player's board.
        - attack strategy, representing who AI is attacking. Can be of 4 types:
                    * 1 - attack random
                    * 2 - attack one that is closed to KO
                    * 3 - attack one that has the highest KOs number
                    * 4 - attack everybody who attacks you. Theonly option that allows attacking several
                          players at once.

    In the default setting, the board looks like this (newly spawned piece is denoted with x-s)

    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  x  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  x  x  0  0 10 10 10   this part of the board can not be seen by players
    10 10 10  0  0  0  0  0  0  x  0  0  0 10 10 10   pieces spawn there to avoid conflicts at spawn
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    -----------------------------------------------   if pieces cramped up in the board cross this line, the game is over
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10   the game happens there
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10   boards can be shown or hidden, this is optional
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10  0  0  0  0  0  0  0  0  0  0 10 10 10
    10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
    10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
    """

    def __init__(self):
        # spawn current piece of a player
        self.piece_current = Piece()
        # spawn a swap piece
        self.piece_swap = Piece()
        # spawn queue
        self.piece_queue =[Piece() for _ in range(5)]
        # initialize board
        self.board = np.zeros((BOARD_HEIGHT + 8, BOARD_WIDTH + 6), dtype=np.int16)
        # set up borders
        self.board[:, 0:3] = 10
        self.board[BOARD_HEIGHT+5:BOARD_HEIGHT+8, :] = 10
        self.board[:, BOARD_WIDTH+3:BOARD_WIDTH+6] = 10
        # init the number of players kicked out by this particular one
        self.KOs = 0
        # init place in the leaderboard
        self.place = None
        # stores the number of steps until post
        self.incoming_garbage = []
        # init attack strategy
        # out of 0, 1, 2, 3
        self.attack_strategy = 1


class State99:
    """
    Class holding all information necessary for the game. It is also able to produce observation, needed for AI
    to make decisions
    """

    def __init__(self, num_players):
        # queue used to store all events of player-to-player interactions, such as attacks with garbage
        self.event_queue = []
        # the list which holds information about all players.
        self.players = [Player99() for _ in range(num_players)]


    def observe(self, player_id):
        """
        function to produce information needed for agent to make decision

        :param int player_id:
        :return: fill when decide what to observe
        """
        pass