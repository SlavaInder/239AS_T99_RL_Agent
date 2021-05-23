import gym
from gym import error, spaces, utils
from gym.utils import seeding
from numpy.lib.function_base import _parse_gufunc_signature
from numpy.random import poisson
from .state import *
from .renderers import Renderer
from copy import copy, deepcopy
import pygame
from os import environ
# Silence the pygame printing
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class T99SC(gym.Env):
    """
    public API consists only of 4 funcs:
        - step
        - reset
        - render
        - close

    Note: this T99 environment provides Simplified Control for RL agent
    """
    metadata = {'render.modes': ['human', 'debug']}

    """            Public Methods here             """

    def __init__(self, enemy, num_players=2):
        """
        This is a simplified T99 environment necessary for

        :param enemy: a strategy able to produce actions based on observation; note, that this is not the strategy
                                    we are training, instead, this is a strategy we are training against. Usually,
                                    this is the previous iteration of AI
        :param num_players: number of competing agents (ideally, 99) in the game
        """
        self.action_space = {
            column: np.arange(10),
            rotation: np.arange(3),
            attack: np.arange(4) - 1,
            swap: [True, False]
        }
        self.enemy = enemy
        self.state = State99(num_players)
        # an array to keep track of who is in the game
        self.active_players = np.ones(num_players).astype(bool)
        # counter of steps made
        self.current_step = 0
        # variable necessary for proper updates of
        self.pygame_started = False


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

        return next_state, reward, done, info

    def reset(self):
        self.state = State99()


    def render(self, mode='human',show_window=False,image_path="screenshot.png"):
        """
        :param str mode: mode in which rendering works. If debug, returns numpy matrices for each player
        :param bool show_window: assuming mode='human', says whether to show the window or not
        :param str image_path: when mode='human', this is the relative path the screenshot will be saved to
        :return object frame: depending on mode, renders a human-visible screenshot of the game
        """
        if mode == "debug":
            # create a list for players' boards
            frame = []
            # for each player
            for i in range(len(self.state.players)):
                # copy the board together with piece
                temp_board = self._apply_piece(self.state.players[i].board.copy(), self.state.players[i].piece_current)
                # append the list with their board
                frame.append(temp_board)

            # Will let us switch between debug or human mode in same session if desired
            if self.pygame_started:
                self.renderer.quit()
                self.pygame_started = False
            
        elif mode == "human":
            
            # get a copy of the current state
            temp_state = deepcopy(self.state)
            # apply the piece to each board
            for i in range(len(temp_state.players)):
                # copy the board together with piece
                temp_state.players[i].board = self._apply_piece(temp_state.players[i].board.copy(), temp_state.players[i].piece_current)

            if not self.pygame_started:
                self.renderer = Renderer(temp_state.players, show_window=show_window)
                self.pygame_started = True
            else:
                self.renderer.update_state(temp_state.players)

            self.renderer.draw_screen()
            self.renderer.save_screen_as_image(image_path)
            
            frame = None

        return frame

    def close(self):
        if self.pygame_started:
            self.renderer.quit()

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
        if action in [1, 2, 3, 4]:
            self.state.players[player_id].attack_strategy = action
        # swap a piece
        # note that after a piece was swapped, new piece starts at the top to avoid conflicts at collisions
        # this means that after a piece was placed, swapped piece corrdinates return to the top
        if action == 5:
            self.state.players[player_id].piece_current, self.state.players[player_id].piece_swap = \
                self.state.players[player_id].piece_swap, self.state.players[player_id].piece_current
        # Move piece left
        if action == 6:
            success = self._move(self.state.players[player_id].board,
                                 self.state.players[player_id].piece_current,
                                 -1, 0)
        # Move piece right
        elif action == 7:
            success = self._move(self.state.players[player_id].board,
                                 self.state.players[player_id].piece_current,
                                 1, 0)
        # Move piece clockwise 90 degrees
        elif action == 8:
            self._rotate_piece(self.state.players[player_id].board,
                               self.state.players[player_id].piece_current,
                               clockwise=True)

        # Move piece counter clockwise 90 degrees
        elif action == 9:
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
            self.active_players[player_id].place = position
            # if so, update the list of active players
            self.active_players[player_id] = False

    """         Helper functions here           """

    def _apply(self, board, piece):
        # stick piece to the board, and return new board
        board[piece.y-2:piece.y+3, piece.x-2:piece.x+3] += piece.matrix
        return board

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
                return_state.append((self.state.players[i].board,
                    self.state.players[i].piece_swap,
                    self.state.players[i].KOs,
                    self.state.players[i].incoming_garbage,
                    self.state.players[i].place,
                    self.state.players[i].attack_strategy))
            # Otherwise return only the board and the number of badges (number of KOs in our case).
            else:
                return_state.append((self.state.players[i].board, self.state.players[i].KOs))

        return return_state