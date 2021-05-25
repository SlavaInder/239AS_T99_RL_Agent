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
    """
    metadata = {'render.modes': ['human', 'debug']}
    settings = {
        "attack_delay": 5,      # after attack_delay number of steps expires, garbage moves from the queue to the board
        "r_clear_line": 0.01,   # the reward for clearing one line
        "r_send_line": 0.02,    # the reward for sending one line of garbage
        "r_win": 1000           # the reward for winning
    }

    def __init__(self, enemy, num_players=2):
        """
        This class is a modification of standard SC environment controlled by raw button presses. Instead, it uses the
        desired end state as the action. Observation of the state includes all possible configurations for the next step
        :param enemy: a strategy able to produce actions based on observation; note, that this is not the strategy
                                    we are training, instead, this is a strategy we are training against. Usually,
                                    this is the previous iteration of AI
        :param num_players: number of competing agents (ideally, 99) in the game
        """
        # initialize AI for NPC-s
        self.enemy = enemy
        # initialize state
        self.state = State99(num_players)
        # an array to keep track of who is in the game
        self.active_players = np.ones(num_players).astype(bool)
        # counter of steps made
        self.current_step = 0
        # variable necessary for proper updates of
        self.pygame_started = False
        # pygame renderer
        self.renderer = None

    """            Public Methods here             """

    def step(self, action):
        """
        the function which makes one update of the environment.

        :param dict action: a dictionary with reward, state, and finishing status.
        :return object observation: a tuple with a list of all possible next states and list of corresponding rewards
        :return float reward: amount of reward achieved by the previous action
        :return boolean done: whether it’s time to reset the environment again, True if the agent loses or wins
        :return dict info: diagnostic information useful for debugging.
        """
        # save the reward
        reward = action["reward"]
        self.state.players[0].score += reward
        # empty variable for debug
        info = {}

        # process current step
        # init array for states the agent and npc-s will end up after the step finishes
        next_states = [None for _ in range(len(self.state.players))]
        # register the agent's step and event queue it created
        next_states[0] = action["state"].players[0]
        self.state.event_queue.extend(action["state"].event_queue)
        # process other player's moves
        for i in range(1, len(self.state.players)):
            # if the player is active
            if self.active_players[i]:
                # observe which action an npc can take
                npc_options, _ = self._observe(i)
                # choose the best action
                npc_action = self.enemy.action(npc_options)
                # register the npc's step and event queue it created
                next_states[i] = npc_action.players[i]
                self.state.event_queue.extend(npc_action.event_queue)

        # update game's state
        self.state.players = next_states
        # distribute garbage
        self._process_event_queue()
        # check who lost during time step
        self._check_kos()
        # if the agent has lost, stop the game
        if not self.active_players[0]:
            done = True
        else:
            done = False

        # if game continues
        if not done:
            # calculate possible next states
            observation = self._observe(0)
        else:
            # or return empty tuple
            observation = ([], [])

        return observation, reward, done, info

    def reset(self):
        self.state = State99(num_players=len(self.active_players))
        self.active_players = np.ones(len(self.active_players)).astype(bool)

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

    """         Helper functions here           """

    def _observe(self, player_id):
        """
        this function returns all possible states and rewards the agent can end up after doing one step
        this function first aligns the current and swap pieces horizontally;  then
        for each piece (current / swap)
            for each rotation (0 / 90 / 180 / 270 degrees)
                it rotates the piece
                for each delta x (-5, -4 ... 3, 4)
                    it moves it by delta x or as far as possible, then drops as far as possible.
                    after that, garbage is added, rows are cleared, reward assigned, garbage is sent, reward assigned,
        """
        # init the tuple of options, holding next_states and corresponding rewards
        options = ([], [])
        start_state = deepcopy(self.state)
        # align current piece and swap piece
        start_state.players[player_id].piece_current.x = start_state.players[player_id].board.shape[1] // 2
        start_state.players[player_id].piece_swap.x = start_state.players[player_id].board.shape[1] // 2
        # add garbage
        self._apply_garbage(start_state.players[player_id])
        # check options for each piece
        for piece in [start_state.players[player_id].piece_current, start_state.players[player_id].piece_swap]:
            # for each possible rotation
            for _ in range(4):
                self._rotate_piece(start_state.players[player_id].board, piece)
                # for each possible x coordinate
                for dx in np.arange(10) - 5:
                    # check if the coordinate is valid
                    valid = self._move(start_state.players[player_id].board, piece, dx, 0)
                    if valid:
                        # if the coordinate is valid, drop the piece until it is stuck
                        stuck = False
                        while not stuck:
                            stuck = not self._move(start_state.players[player_id].board, piece, 0, 1)
                        # init a copy of state and a reward
                        reward = 0
                        end_state = deepcopy(start_state)
                        # apply piece to the copy
                        end_state.players[player_id].board = self._apply_piece(end_state.players[player_id].board,
                                                                               piece)
                        # clear lines
                        end_state.players[player_id].board, num_lines = \
                            self._clear_rows(end_state.players[player_id].board)
                        # get rewarded
                        reward +=  1 + (num_lines ** 2) * self.state.players[player_id].board.shape[1]

                        end_state.players[player_id].num_lines_cleared = num_lines
                        # push attacks to the queue
                        pass
                        # get rewarded
                        pass
                        # depending on whether the piece was current or swap, update piece queue
                        if np.all(piece.matrix == end_state.players[player_id].piece_current.matrix):
                            self._next_piece(end_state.players[player_id])
                        else:
                            # else swap piece and then update piece queue
                            end_state.players[player_id].piece_current, end_state.players[player_id].piece_swap = \
                                end_state.players[player_id].piece_swap, end_state.players[player_id].piece_current
                            self._next_piece(end_state.players[player_id])
                        # append copy and reward to options
                        options[0].append(end_state)
                        options[1].append(reward)

                    # return piece to original place
                    piece.x = start_state.players[player_id].board.shape[1] // 2
                    piece.y = 2

        return options

    def _process_event_queue(self):
        # function that processes the attacks player's send to each other in the form of events
        for event in self.state.event_queue:
            # send num_lines lines to the target
            for i in range(event.num_lines):
                self.state.players[event.target].incoming_garbage.append(T99SC.settings["attack_delay"])

    def _check_kos(self):
        b_height, b_width = self.state.players[0].board.shape
        # function that updates the list of active players
        for i in range(len(self.state.players)):
            # if the player was kicked out and assigned a place
            if self.state.players[i].place is not None:
                self.active_players[i] = False
            # if the player has their board full
            elif np.sum(self.state.players[i].board.astype(bool)[0:5, 3:b_width - 3]) > 0:
                # assign the position in the leaderboard
                position = len(self.active_players) - np.sum(np.where(self.active_players is True, 1, 0))
                self.state.players[i].place = position
                # if so, update the list of active players
                self.active_players[i] = False

    def _apply_piece(self, board, piece):
        # stick piece to the board, and return new board
        board[piece.y-2:piece.y+3, piece.x-2:piece.x+3] += piece.matrix
        return board

    def _apply_garbage(self, player):
        # adds garbage to a player's board and updates expiration time in garbage queue
        # calculate board's width
        b_height, b_width = player.board.shape
        # init total number of lines that need to be sent
        total_lines = 0
        # update counter for everything in the queue
        for i in range(len(player.incoming_garbage)):
            player.incoming_garbage[i] -= 1
            # if line's time is up
            if player.incoming_garbage[i] <= 0:
                # add it to the counter
                total_lines += 1
        # remove expired lines
        player.incoming_garbage = [g for g in player.incoming_garbage if g > 0]
        # choose x that will miss from the garbage
        missing_x = np.random.choice(np.arange(10))
        # update player's board
        # first move all existing lines to the top by "total_lines" lines
        player.board[0:25 - total_lines, 3:b_width - 3] = player.board[total_lines:25, 3:b_width - 3]
        # then clear free space
        player.board[25 - total_lines:25, 3:b_width - 3] = 0
        # then fill free space with garbage
        player.board[25 - total_lines:25, 3:3 + missing_x] = 8
        player.board[25 - total_lines:25, 4 + missing_x:b_width - 3] = 8

    def _clear_rows(self, board):
        b_height, b_width = board.shape
        # check which lines are cleared
        cleared = np.prod(board.astype(bool), axis=1)
        # save the number of lines cleared to calculate attack power
        attack = np.sum(cleared) - 3
        # for each cleared line
        i = len(cleared) - 4
        while i > 4:
            # if the line needs to be cleared
            if cleared[i] > 0:
                # clear the line
                board[i, 3:b_width - 3] = 0
                cleared[i] = 0
                # shift all lines from the top by 1
                board[6:i + 1, 3:b_width - 3] = board[5:i, 3:b_width - 3]
                cleared[6:i + 1] = cleared[5:i]
                # clear the top line, which does not have pieces after shift
                board[5, 3:b_width - 3] = 0
                cleared[5] = 0
            else:
                i -= 1

        return board, attack

    def _collision(self, board, piece):
        # check whether at least one element of the piece overlaps wit board
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
