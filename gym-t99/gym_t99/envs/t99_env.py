import gym
from gym import error, spaces, utils
from gym.utils import seeding
from numpy.lib.function_base import _parse_gufunc_signature
from numpy.random import poisson
from .state import *

from copy import copy, deepcopy

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" #Silence the pygame printing
import pygame


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
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.enemy = enemy
        self.state = State99(num_players)
        # an array to keep track of who is in the game
        self.active_players = np.ones(num_players).astype(bool)
        # counter of steps made
        self.current_step = 0
        # how many moves per environment update a player can do
        self.update_frequency = 3

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
        if mode=="debug":
            # create a list for players' boards
            frame = []
            # for each player
            for i in range(len(self.state.players)):
                # copy the board together with piece
                temp_board = self._apply_piece(self.state.players[i].board.copy(), self.state.players[i].piece_current)
                # append the list with their board
                frame.append(temp_board)
            
            if self.pygame_started: #Will let us switch between debug or human mode in same session if desired
                self.renderer.quit()
                self.pygame_started = False
            
        elif mode == "human":
            # TODO: we need to apply a piece to a board before rendering
            # however, application can be done only to the copy of a player's state
            # (because of other game logic, we can not change self.state)


            # get a copy of the current state
            temp_state = deepcopy(self.state)
            # apply the piece to each board
            for i in range(len(temp_state.players)):
                # copy the board together with piece
                temp_state.players[i].board = self._apply_piece(temp_state.players[i].board.copy(), temp_state.players[i].piece_current)

            if not self.pygame_started:
                self.renderer = Renderer(temp_state.players, show_window=show_window)
                self.pygame_started = True
            
            self.renderer.draw_screen()
            self.renderer.save_screen_as_image(image_path)
            
            frame=None

        return frame

    def close(self):
        if self.pygame_started:
            self.renderer.quit()
            
    def _apply(self, board, piece):
        # stick piece to the board, and return new board
        board[piece.y-2:piece.y+3, piece.x-2:piece.x+3] += piece.matrix
        return board

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

class Renderer():
    '''
    Handles the screen rendering
    '''
    #Colors
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    ORANGE = (255, 127, 0)
    CYAN = (0, 183, 235)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)
    GREY = (90, 90, 90)
    PINK = (231,7, 247)
    TRANS_WHITE = (255, 255, 255, 50)

    #Piece to Colors Map, from : https://tetris.fandom.com/wiki/Tetromino
    piece_colors = {
        0 : BLACK, 
        1 : CYAN,
        2 : BLUE,
        3 : ORANGE,
        4 : MAGENTA,
        5 : RED,
        6 : GREEN,
        7 : YELLOW,
        10 : GREY
    }

    
    def __init__(self,players,show_window=False,gridsize=50):
        '''
        params:
            players : list of type Player99, should be all but the main player, each of their boards will be drawn, with the first player drawn in the middle and larger
            show_window : if true, will show the GUI in a window
            gridsize : int, is the size of each block in the primary board
        '''

        #Set players
        self.player = players[0]
        self.npcs = players[1:]

        #Get parameters of board
        self.BOARD_HEIGHT, self.BOARD_WIDTH = players[0].board.shape #Get dimensions of board
        self.top_rows_ignore = 5 #How many rows on the top to ignore
        self.num_walls_side, self.num_walls_bottom = Renderer.get_wall_sizes(self.player.board) #How many walls to ignore. We do this because walls can clutter
        self.BOARD_HEIGHT = self.BOARD_HEIGHT  - self.top_rows_ignore - self.num_walls_bottom
        self.BOARD_WIDTH = self.BOARD_WIDTH - 2 * self.num_walls_side

        #Set pixel constants of board
        self.MAIN_BOARD_GRIDSIZE = gridsize #The only constant here, the pixel size of the main board
        self.SECONDARY_BOARD_GRIDSIZE = max(1,int(self.MAIN_BOARD_GRIDSIZE // 7)) #Gridsize of smaller boards
        self.SMALL_PADDING_PX, self.MEDIUM_PADDING_PX, self.LARGE_PADDING_PX = 2 * self.SECONDARY_BOARD_GRIDSIZE, 6 * self.SECONDARY_BOARD_GRIDSIZE, self.MAIN_BOARD_GRIDSIZE
        self.VERT_PADDING_PX, self.SIDE_PADDING_PX = self.LARGE_PADDING_PX, self.SMALL_PADDING_PX

        #Set widths of important areas
        self.NPC_WIDTH_PX = 7 * self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_WIDTH + 6 * self.SMALL_PADDING_PX #Width of single side of NPCS
        self.NPC_HEIGHT_PX = 7 * self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_HEIGHT + 6 * self.MEDIUM_PADDING_PX #Width of single side of NPCS
        
        swap_width = int(self.MAIN_BOARD_GRIDSIZE // 2) * 5 
        queue_width = int(self.MAIN_BOARD_GRIDSIZE // 2) * 5 
        self.PLAYER_WIDTH_PX = swap_width + self.MAIN_BOARD_GRIDSIZE * self.BOARD_WIDTH + queue_width #Left swap area + board size + right next pieces + right queue piece area
        
        #Set full dimensions of the board
        self.FULL_WIDTH_PX = 2 * self.SIDE_PADDING_PX + 2 * self.MEDIUM_PADDING_PX + 2 * self.NPC_WIDTH_PX + self.PLAYER_WIDTH_PX
        self.FULL_HEIGHT_PX = 2 * self.VERT_PADDING_PX + self.NPC_HEIGHT_PX
        
        #Create the window
        #----------------
        if not show_window:
            environ["SDL_VIDEODRIVER"] = "dummy" #Makes window not appear
        
        pygame.init()
        pygame.display.set_caption('Tetris')
        self.window = pygame.display.set_mode((self.FULL_WIDTH_PX , self.FULL_HEIGHT_PX ))
        
        #Create the NPC Boards
        #---------------------
        self.npc_board_renderers = [] 
        
        #Left half
        x_init = self.SIDE_PADDING_PX
        y_init = self.VERT_PADDING_PX
        x_step = self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_WIDTH + self.SMALL_PADDING_PX
        y_step = self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_HEIGHT + self.MEDIUM_PADDING_PX
        n = 0
        for x in range(7):
            for y in range(7):
                if n >= len(self.npcs):
                    continue
                x_orig = x_init + x_step * x
                y_orig = y_init + y_step * y
                new_board = BoardRenderer(self.npcs[n], x_orig, y_orig, self.SECONDARY_BOARD_GRIDSIZE, self.window, self.top_rows_ignore, self.num_walls_side, self.num_walls_bottom)
                self.npc_board_renderers.append(new_board)
                n += 1

        #Right half
        x_init = self.FULL_WIDTH_PX - self.NPC_WIDTH_PX - self.SIDE_PADDING_PX
        y_init = self.VERT_PADDING_PX
        x_step = self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_WIDTH + self.SMALL_PADDING_PX
        y_step = self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_HEIGHT + self.MEDIUM_PADDING_PX
        for x in range(7):
            for y in range(7):
                if n >= len(self.npcs):
                    continue
                x_orig = x_init + x_step * x
                y_orig = y_init + y_step * y
                new_board = BoardRenderer(self.npcs[n], x_orig, y_orig, self.SECONDARY_BOARD_GRIDSIZE, self.window, self.top_rows_ignore, self.num_walls_side, self.num_walls_bottom)
                self.npc_board_renderers.append(new_board)
                n += 1

        #Create the Players Board
        #-----------------------
        x_orig = self.SIDE_PADDING_PX + self.NPC_WIDTH_PX + self.MEDIUM_PADDING_PX + swap_width
        y_orig = self.VERT_PADDING_PX + int(self.MAIN_BOARD_GRIDSIZE * 1.5)
        self.player_board = PlayerRenderer(self.player,x_orig,y_orig,self.MAIN_BOARD_GRIDSIZE,self.window, self.top_rows_ignore, self.num_walls_side, self.num_walls_bottom)

    def set_player_position(self):
        '''
        Sets the players board to the lowest of the NPC's - 1. Or 100 if no NPC's have positions
        '''
        # Iterate through all boards
        # If no integer found, then set position equal to None
        # If there is a minimum integer found, then set self.player_board to 1 minus that
        
        self.player_board.num_total_players = len(self.npc_board_renderers) + 1

        if self.player.place != None:
            self.player_board.position = self.player.place
            return

        min_position = len(self.npc_board_renderers) + 2
        for board in self.npc_board_renderers:
            npc = board.player
            position = npc.place
            if position == None:
                continue
            if position < min_position:
                min_position = position
        
        self.player_board.position = min_position - 1


    @staticmethod
    def get_wall_sizes(board):
        '''
        Takes a board, and returns a tuple of the number of walls on the sides and on the bottom
        '''
        side_num = 0
        i = 0
        while board[int(board.shape[0]/2),i] == 10:
            side_num += 1
            i += 1

        i = board.shape[0]-1
        bot_num = 0
        while board[i,int(board.shape[1]/2)] == 10:
            bot_num += 1
            i -= 1
        return side_num, bot_num 
    
    def draw_screen(self):
        '''
        Draws the screen
        '''
        self.set_player_position() #Do a custom calculation of our player's position 
        #self.window.fill(Renderer.YELLOW)
        fill_gradient(self.window, Renderer.CYAN, Renderer.MAGENTA, rect=None, vertical=True, forward=True)
        
        #Draw NPC boards
        for board_renderer in self.npc_board_renderers:
            board_renderer.draw_board()
        
        #Draw user's board
        self.player_board.draw_board()
        pygame.display.update()


    
    def save_screen_as_image(self,path : str):
        '''
        Saves the screen as an image, will update this in the future to meet needs of learning
        '''
        pygame.image.save(self.window, path)

    def quit(self):
        '''
        Quits pygame
        Not working on my machine. No fix known.
        '''
        pygame.display.quit()
        pygame.quit()
    
class BoardRenderer():
        '''
        Handles the rendering of each individual board
        '''
        def __init__(self, player, origin_x_px, origin_y_px, gridsize, window, top_rows_ignore, num_walls_side, num_walls_bottom):
            '''
            params:
                board : 2D np matrix like described in Player99 class
                origin_x_px : int, origin of this board x in pixels
                origin_y_px : int, origin of this board y in pixels
                gridsize : int, pixel size of each block
                window : the pygame window which to draw on
                top_rows_ignore : int, number of rows on top to not draw
                num_walls_side : int, number of rows on side, we will not be drawing these
                num_walls_bottom : int, number of rows on bottom, we will not be drawing these
            '''
            self.player = player
            self.window = window
            self.board = player.board
            self.top_rows_ignore = top_rows_ignore
            self.num_walls_side , self.num_walls_bottom = num_walls_side, num_walls_bottom
            self.GRIDSIZE = gridsize
            self.BOARD_HEIGHT, self.BOARD_WIDTH = self.board.shape #Get dimensions of board
            self.BOARD_HEIGHT = self.BOARD_HEIGHT - self.top_rows_ignore - self.num_walls_bottom #Ignore top 5 rows
            self.BOARD_WIDTH -= 2 * self.num_walls_side
            self.BOARD_HEIGHT_PX = self.BOARD_HEIGHT * self.GRIDSIZE
            self.BOARD_WIDTH_PX = self.BOARD_WIDTH * self.GRIDSIZE
            self.SHADOW_SIZE_PX = max(1,int(self.GRIDSIZE // 15))
            self.LINE_WIDTH = max(2,int(gridsize // 15))
            self.GRID_WIDTH = max(1,int(gridsize // 30))
            self.ORIGIN_X_PX = origin_x_px
            self.ORIGIN_Y_PX = origin_y_px
            
            self.OUTLINE_CLR = Renderer.WHITE
            self.GRID_CLR = Renderer.GREY
            self.fontsize_ko = int(self.GRIDSIZE * 3.9)
            self.arial_ko = pygame.font.SysFont('Arial Black', self.fontsize_ko)


        def draw_board(self):
            '''
            Draws a board
            '''
            self.draw_background()
            self.draw_outline()
            if self.player.place != None:
                self.draw_elimination()
            else:
                self.draw_all_blocks()
        
        def draw_background(self):
            '''
            Draws black background for board
            '''
            pygame.draw.rect(self.window, Renderer.BLACK, (self.ORIGIN_X_PX+1, self.ORIGIN_Y_PX+1, self.BOARD_WIDTH_PX+1, self.BOARD_HEIGHT_PX+1), 0)
        
        def draw_elimination(self):
            string1 = "KO"
            if len(str(self.player.place)) == 1:
                string2 = str(self.player.place)
            else:
                string2 = str(self.player.place)
            place_text = self.arial_ko.render(string1, True, Renderer.WHITE)
            place_text2 = self.arial_ko.render(string2, True, Renderer.WHITE)
            

            center_x = int(self.ORIGIN_X_PX + self.GRIDSIZE * self.BOARD_WIDTH * 0.5)
            center_y = int(self.ORIGIN_Y_PX + self.GRIDSIZE * self.BOARD_HEIGHT * 0.35)
            center_y_2 = int(self.ORIGIN_Y_PX + self.GRIDSIZE * self.BOARD_HEIGHT * 0.35 + self.fontsize_ko*1.2)
            text_rect_1 = place_text.get_rect(center=(center_x, center_y))
            text_rect_2 = place_text2.get_rect(center=(center_x, center_y_2))
            self.window.blit(place_text,text_rect_1)
            self.window.blit(place_text2,text_rect_2)
        
        def draw_outline(self):
            top_left = (self.ORIGIN_X_PX - 1, self.ORIGIN_Y_PX - 1)
            top_right = (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX + 1, self.ORIGIN_Y_PX-1)
            bottom_left = (self.ORIGIN_X_PX - 1, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX + 1)
            bottom_right = (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX+1, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX + 1)
            pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, bottom_left, self.LINE_WIDTH)
            pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, top_right, self.LINE_WIDTH)
            pygame.draw.line(self.window, self.OUTLINE_CLR, top_right, bottom_right, self.LINE_WIDTH)
            pygame.draw.line(self.window, self.OUTLINE_CLR, bottom_left, bottom_right, self.LINE_WIDTH)

        def draw_grid(self):
            '''
            Draws the background grid on the screen for the pieces
            '''
            for i in range(1,self.BOARD_WIDTH):
                pygame.draw.line(self.window, self.GRID_CLR, (self.ORIGIN_X_PX + i * self.GRIDSIZE, self.ORIGIN_Y_PX), (self.ORIGIN_X_PX + i * self.GRIDSIZE, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX), self.GRID_WIDTH)

            for i in range(1,self.BOARD_HEIGHT):
                pygame.draw.line(self.window, self.GRID_CLR, (self.ORIGIN_X_PX, self.ORIGIN_Y_PX + i * self.GRIDSIZE), (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX , self.ORIGIN_Y_PX + i * self.GRIDSIZE), self.GRID_WIDTH)

        def draw_all_blocks(self):
            '''
            Draws all blocks, careful to ignore the rows which we do not want to draw
            '''
            for row in range(self.top_rows_ignore,self.board.shape[0]-self.num_walls_bottom): 
                for col in range(self.num_walls_side,self.board.shape[1]-self.num_walls_side):
                    self.draw_block(col-self.num_walls_side,row-self.top_rows_ignore,self.board[row][col])
        
        def draw_block(self,grid_x,grid_y,piece_int):
            '''
            Takes a grid_x (column) and a grid_y (row), and the int of the piece
            Draws the piece onto the window
            '''
            if piece_int == 0: #Skip empty squares
                return 
            x_px = grid_x * self.GRIDSIZE + self.ORIGIN_X_PX
            y_px = grid_y * self.GRIDSIZE + self.ORIGIN_Y_PX
            clr = Renderer.piece_colors[piece_int]
            pygame.draw.rect(self.window, clr, (x_px+1, y_px+1, self.GRIDSIZE, self.GRIDSIZE), 0)
            
           
            # if piece_int == 0: #Skip empty squares
            #     return 
            # x_px = grid_x * self.GRIDSIZE + self.ORIGIN_X_PX
            # y_px = grid_y * self.GRIDSIZE + self.ORIGIN_Y_PX
            # clr = Renderer.piece_colors[piece_int]
            # #pygame.draw.rect(self.window, clr, (x_px, y_px, self.GRIDSIZE-(2*self.SHADOW_SIZE_PX), self.GRIDSIZE-(2*self.SHADOW_SIZE_PX)), 0)
            # pygame.draw.rect(self.window, clr, (x_px+1, y_px+1, self.GRIDSIZE+1, self.GRIDSIZE+1), 0)
            
            # #if piece_int != 10: #Add highlight around all non-grey pieces
            # #    pygame.draw.rect(self.window, Renderer.WHITE, (x_px, y_px, self.GRIDSIZE, self.GRIDSIZE), self.SHADOW_SIZE_PX) #Outline

class PlayerRenderer(BoardRenderer):

    def __init__(self, player, origin_x_px, origin_y_px, gridsize, window, top_rows_ignore, num_walls_side, num_walls_bottom):
            '''
            params:
                board : 2D np matrix like described in Player99 class
                origin_x_px : int, origin of this board x in pixels
                origin_y_px : int, origin of this board y in pixels
                gridsize : int, pixel size of each block
                window : the pygame window which to draw on
                top_rows_ignore : int, number of rows on top to not draw
                num_walls_side : int, number of rows on side, we will not be drawing these
                num_walls_bottom : int, number of rows on bottom, we will not be drawing these
            '''
            super().__init__(player, origin_x_px, origin_y_px, gridsize, window, top_rows_ignore, num_walls_side, num_walls_bottom)
            self.HALF_GRIDSIZE = int(self.GRIDSIZE // 2)

            self.position = None
            self.num_total_players = None

            #QUEUE PIECE
            #----------
            #Dims: 33 x 5 of half gridsize
            self.GRIDS_PER_PIECE_VERT = 5
            self.QUEUE_WIDTH_PX = self.HALF_GRIDSIZE * 5
            self.QUEUE_HEIGHT_PX = 3 * self.HALF_GRIDSIZE + self.GRIDS_PER_PIECE_VERT * 5 * self.HALF_GRIDSIZE
            self.QUEUE_X_PX = self.ORIGIN_X_PX + self.BOARD_WIDTH * self.GRIDSIZE
            self.QUEUE_Y_PX = self.ORIGIN_Y_PX

            #PLACE AREA
            #---------
            #Dims: 5 x (self.BOARD_HEIGHT - self.QUEUE_HEIGHT)
            self.PLACE_WIDTH_PX = self.QUEUE_WIDTH_PX
            self.PLACE_HEIGHT_PX = self.BOARD_HEIGHT_PX - self.QUEUE_HEIGHT_PX
            self.PLACE_X_PX = self.QUEUE_X_PX
            self.PLACE_Y_PX = self.ORIGIN_Y_PX + self.QUEUE_HEIGHT_PX + 1


            #SWAP PIECE
            #---------
            #Dims: 8 x 5 of half gridsize
            #Vars for swap piece box
            #Defines top left area
            self.SWAP_WIDTH_PX = self.HALF_GRIDSIZE * 5
            self.SWAP_HEIGHT_PX = 8 * self.HALF_GRIDSIZE
            self.SWAP_X_PX = self.ORIGIN_X_PX - self.SWAP_WIDTH_PX
            self.SWAP_Y_PX = self.ORIGIN_Y_PX
            

            #GARBAGE PIECE AREA
            #---------
            #Dims: (self.BOARD_HEIGHT - 4) x 2 of normal gridsize
            self.GARBAGE_HEIGHT = self.BOARD_HEIGHT - 4 #Useful to see if we can fit pieces
            self.GARBAGE_WIDTH_PX = int(self.GRIDSIZE * 1.5)
            self.GARBAGE_OFFSET_X = int(self.GRIDSIZE * 0.25)
            self.GARBAGE_HEIGHT_PX = self.GARBAGE_HEIGHT * self.GRIDSIZE
            self.GARBAGE_X_PX = self.ORIGIN_X_PX - self.GARBAGE_WIDTH_PX
            self.GARBAGE_Y_PX = self.ORIGIN_Y_PX + self.SWAP_HEIGHT_PX

            self.OUTLINE_CLR = Renderer.WHITE

            #Fontsizes
            self.fontsize_swap = int(self.GRIDSIZE / 2)
            self.fontsize_place = int(self.GRIDSIZE / 1.4)
            self.fontsize_place_small = int(self.GRIDSIZE / 1.7)
            self.fontsize_target = int(self.GRIDSIZE)

            #Fonts
            self.arial_target = pygame.font.SysFont('Arial Black', self.fontsize_target)
            
            self.arial_headers = pygame.font.SysFont('Arial Black', self.fontsize_swap)
            self.arial_place_small = pygame.font.SysFont('Arial Black', self.fontsize_place_small)
            self.arial_place = pygame.font.SysFont('Arial Black', self.fontsize_place)
            

            #TODO: 
            # - Different sick looking background color, maybe a gradient??
            # - Figure out screenshot system
            
    def draw_board(self):
            '''
            Draws a board
            '''
            self.draw_background()
            self.draw_target()
            self.draw_swap() 
            self.draw_place() 
            self.draw_queue()
            self.draw_garbage()
            
            if self.player.place != None:
                self.draw_elimination()
                self.draw_outline()
            else:
                self.draw_grid()
                self.draw_all_blocks()
                self.draw_outline()
    
    def draw_outline(self):
            top_left = (self.ORIGIN_X_PX - 1, self.ORIGIN_Y_PX - 1)
            top_right = (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX, self.ORIGIN_Y_PX-1)
            bottom_left = (self.ORIGIN_X_PX - 1, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX)
            bottom_right = (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX)
            pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, bottom_left, self.LINE_WIDTH)
            pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, top_right, self.LINE_WIDTH)
            pygame.draw.line(self.window, self.OUTLINE_CLR, top_right, bottom_right, self.LINE_WIDTH)
            pygame.draw.line(self.window, self.OUTLINE_CLR, bottom_left, bottom_right, self.LINE_WIDTH)
            
    def draw_block(self,grid_x,grid_y,piece_int):
            '''
            Takes a grid_x (column) and a grid_y (row), and the int of the piece
            Draws the piece onto the window
            '''
            if piece_int == 0: #Skip empty squares
                return 
            x_px = grid_x * self.GRIDSIZE + self.ORIGIN_X_PX
            y_px = grid_y * self.GRIDSIZE + self.ORIGIN_Y_PX
            clr = Renderer.piece_colors[piece_int]
            pygame.draw.rect(self.window, clr, (x_px+1, y_px+1, self.GRIDSIZE-(2*self.SHADOW_SIZE_PX)-1, self.GRIDSIZE-(2*self.SHADOW_SIZE_PX)-1), 0)
            
            if piece_int != 10: #Add highlight around all non-grey pieces
                pygame.draw.rect(self.window, Renderer.WHITE, (x_px+1, y_px+1, self.GRIDSIZE-1, self.GRIDSIZE-1), self.SHADOW_SIZE_PX) #Outline 

    def draw_block_custom(self,grid_x, grid_y, grid_orig_x, grid_orig_y, piece_int, gridsize,custom_clr=None):
        if piece_int == 0:
            return
        x_px = grid_x * gridsize + grid_orig_x
        y_px = grid_y * gridsize + grid_orig_y
        if custom_clr == None:
            custom_clr = Renderer.piece_colors[piece_int]
        pygame.draw.rect(self.window, custom_clr, (x_px, y_px, gridsize-(2*self.SHADOW_SIZE_PX), gridsize-(2*self.SHADOW_SIZE_PX)), 0)

        if piece_int != 10: #Add highlight around all non-grey pieces
            pygame.draw.rect(self.window, Renderer.WHITE, (x_px, y_px, gridsize, gridsize), self.SHADOW_SIZE_PX) #Outline 
    
    def draw_target(self):
        mode = [None,"RANDOM","K.O.","LEADER","ATTACKER"]
        text = self.arial_target.render(mode[self.player.attack_strategy], True, Renderer.WHITE)
        center_x = int(self.ORIGIN_X_PX + self.GRIDSIZE * self.BOARD_WIDTH * 0.5)
        center_y = int(self.ORIGIN_Y_PX - self.GRIDSIZE * 0.75)
        text_rect = text.get_rect(center=(center_x, center_y))
        self.window.blit(text,text_rect)

    
    def draw_garbage_piece(self,grid_x, grid_y, grid_orig_x, grid_orig_y, gridsize,clr):
        
        x_px = grid_x * gridsize + grid_orig_x
        y_px = grid_y * gridsize + grid_orig_y
        
        pygame.draw.rect(self.window, clr, (x_px, y_px, gridsize-(2*self.SHADOW_SIZE_PX), gridsize-(2*self.SHADOW_SIZE_PX)), 0)
        pygame.draw.rect(self.window, Renderer.WHITE, (x_px, y_px, gridsize, gridsize), self.SHADOW_SIZE_PX) #Outline 
    
    def draw_place(self):
        #Draw Background rect
        pygame.draw.rect(self.window, Renderer.BLACK, (self.PLACE_X_PX, self.PLACE_Y_PX, self.PLACE_WIDTH_PX, self.PLACE_HEIGHT_PX), 0)
        #Draw outline on top,and bottom
        top_right = (self.PLACE_X_PX + self.PLACE_WIDTH_PX,self.PLACE_Y_PX - 1)
        bottom_left = (self.PLACE_X_PX - 1,self.PLACE_Y_PX + self.PLACE_HEIGHT_PX )
        bottom_right = (self.PLACE_X_PX + self.PLACE_WIDTH_PX ,self.PLACE_Y_PX + self.PLACE_HEIGHT_PX )
        pygame.draw.line(self.window, self.OUTLINE_CLR, top_right, bottom_right, self.LINE_WIDTH) #right
        pygame.draw.line(self.window, self.OUTLINE_CLR, bottom_left, bottom_right, self.LINE_WIDTH) #bottom

        #Draw the place 
        place_str = "{}/{}".format(self.position,self.num_total_players)
        place_text = self.arial_place.render(place_str, True, Renderer.WHITE)

        center_x = int(self.PLACE_X_PX + self.PLACE_WIDTH_PX * 0.5)
        center_y = int(2.3 * self.GRIDSIZE + self.PLACE_Y_PX)
        text_rect = place_text.get_rect(center=(center_x, center_y))

        self.window.blit(place_text,text_rect)

        #Draw the K.O's
        KO_str_1 = "KO's:"
        place_text = self.arial_place_small.render(KO_str_1, True, Renderer.WHITE)

        center_x = int(self.PLACE_X_PX + self.PLACE_WIDTH_PX * 0.5)
        center_y = int(3.7 * self.GRIDSIZE + self.PLACE_Y_PX)
        text_rect = place_text.get_rect(center=(center_x, center_y))

        self.window.blit(place_text,text_rect)


        KO_str_2 = str(self.player.KOs)
        place_text = self.arial_place.render(KO_str_2, True, Renderer.WHITE)

        center_x = int(self.PLACE_X_PX + self.PLACE_WIDTH_PX * 0.5)
        center_y = int(4.5 * self.GRIDSIZE + self.PLACE_Y_PX)
        text_rect = place_text.get_rect(center=(center_x, center_y))

        self.window.blit(place_text,text_rect)



        #Draw the caption
        pygame.draw.line(self.window, self.OUTLINE_CLR, (self.PLACE_X_PX, self.PLACE_Y_PX + 2 * self.HALF_GRIDSIZE), (self.PLACE_X_PX + self.PLACE_WIDTH_PX + 1, self.PLACE_Y_PX + 2 * self.HALF_GRIDSIZE), self.LINE_WIDTH)

        place_text = self.arial_headers.render("PLACE", True, Renderer.WHITE)
        center_x = int(self.PLACE_X_PX + self.PLACE_WIDTH_PX * 0.5)
        center_y = self.HALF_GRIDSIZE + self.PLACE_Y_PX
        text_rect = place_text.get_rect(center=(center_x, center_y))

        self.window.blit(place_text,text_rect)
        


    def draw_queue(self):
        #Draw Background rect
        pygame.draw.rect(self.window, Renderer.BLACK, (self.QUEUE_X_PX, self.QUEUE_Y_PX, self.QUEUE_WIDTH_PX, self.QUEUE_HEIGHT_PX), 0)
        #Draw outline on top,and bottom
        top_left = (self.QUEUE_X_PX - 1,self.QUEUE_Y_PX-1)
        top_right = (self.QUEUE_X_PX + self.QUEUE_WIDTH_PX,self.QUEUE_Y_PX - 1)
        bottom_left = (self.QUEUE_X_PX - 1,self.QUEUE_Y_PX + self.QUEUE_HEIGHT_PX )
        bottom_right = (self.QUEUE_X_PX + self.QUEUE_WIDTH_PX ,self.QUEUE_Y_PX + self.QUEUE_HEIGHT_PX )
        pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, top_right, self.LINE_WIDTH) #top
        pygame.draw.line(self.window, self.OUTLINE_CLR, top_right, bottom_right, self.LINE_WIDTH) #right
        pygame.draw.line(self.window, self.OUTLINE_CLR, (self.QUEUE_X_PX,self.QUEUE_Y_PX + self.QUEUE_HEIGHT_PX ), bottom_right, self.LINE_WIDTH) #bottom

        #Draw the pieces
        offset_y  = 2 * self.HALF_GRIDSIZE
        for i in range(5):
            if i >= len(self.player.piece_queue):
                break
            piece_next = self.player.piece_queue[i].matrix
            
            for i in range(piece_next.shape[0]):
                for j in range(piece_next.shape[1]):
                    self.draw_block_custom(j, i, self.QUEUE_X_PX, self.QUEUE_Y_PX + offset_y, piece_next[i,j],self.HALF_GRIDSIZE)

            offset_y += 5 * self.HALF_GRIDSIZE

        #Draw the caption
        pygame.draw.line(self.window, self.OUTLINE_CLR, (self.QUEUE_X_PX , self.QUEUE_Y_PX + 2 * self.HALF_GRIDSIZE), (self.QUEUE_X_PX + self.QUEUE_WIDTH_PX + 1, self.QUEUE_Y_PX + 2 * self.HALF_GRIDSIZE), self.LINE_WIDTH)

        place_text = self.arial_headers.render("QUEUE", True, Renderer.WHITE)

        center_x = int(self.QUEUE_X_PX + self.QUEUE_WIDTH_PX * 0.5)
        center_y = self.HALF_GRIDSIZE + self.QUEUE_Y_PX
        text_rect = place_text.get_rect(center=(center_x, center_y))

        self.window.blit(place_text,text_rect)



    def draw_swap(self):
        #Draw Background rect
        pygame.draw.rect(self.window, Renderer.BLACK, (self.SWAP_X_PX, self.SWAP_Y_PX, self.SWAP_WIDTH_PX, self.SWAP_HEIGHT_PX), 0)
        #Draw outline on top,and bottom
        top_left = (self.SWAP_X_PX - 1,self.SWAP_Y_PX-1)
        top_right = (self.SWAP_X_PX + self.SWAP_WIDTH_PX,self.SWAP_Y_PX - 1)
        bottom_left = (self.SWAP_X_PX - 1,self.SWAP_Y_PX + self.SWAP_HEIGHT_PX )
        bottom_right = (self.SWAP_X_PX + self.SWAP_WIDTH_PX ,self.SWAP_Y_PX + self.SWAP_HEIGHT_PX )
        pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, top_right, self.LINE_WIDTH) #top
        pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, bottom_left, self.LINE_WIDTH) #left
        pygame.draw.line(self.window, self.OUTLINE_CLR, bottom_left, bottom_right, self.LINE_WIDTH) #bottom

        #Draw the shape
        piece_swap = self.player.piece_swap.matrix #shape : 5x5
        
        for i in range(piece_swap.shape[0]):
            for j in range(piece_swap.shape[1]):
                self.draw_block_custom(j, i, self.SWAP_X_PX, self.SWAP_Y_PX + 2 * self.HALF_GRIDSIZE,piece_swap[i,j],self.HALF_GRIDSIZE)
        
        #Draw the caption
        pygame.draw.line(self.window, self.OUTLINE_CLR, (self.SWAP_X_PX - 1, self.SWAP_Y_PX + 2 * self.HALF_GRIDSIZE), (self.SWAP_X_PX + self.SWAP_WIDTH_PX , self.SWAP_Y_PX + 2 * self.HALF_GRIDSIZE), self.LINE_WIDTH)

        place_text = self.arial_headers.render("SWAP", True, Renderer.WHITE)

        center_x = int(self.SWAP_X_PX + self.SWAP_WIDTH_PX * 0.5)
        center_y = self.HALF_GRIDSIZE + self.SWAP_Y_PX
        text_rect = place_text.get_rect(center=(center_x, center_y))

        self.window.blit(place_text,text_rect)

    def draw_garbage(self):
        #Draw Background rect
        pygame.draw.rect(self.window, Renderer.BLACK, (self.GARBAGE_X_PX, self.GARBAGE_Y_PX + self.LINE_WIDTH, self.GARBAGE_WIDTH_PX, self.GARBAGE_HEIGHT_PX-self.LINE_WIDTH), 0)
        #Draw Outline on left and bottom
        top_left = (self.GARBAGE_X_PX - 1,self.GARBAGE_Y_PX)
        bottom_left = (self.GARBAGE_X_PX - 1,self.GARBAGE_Y_PX + self.GARBAGE_HEIGHT_PX + 1)
        bottom_right = (self.GARBAGE_X_PX + self.GARBAGE_WIDTH_PX + 1,self.GARBAGE_Y_PX + self.GARBAGE_HEIGHT_PX + 1)
        pygame.draw.line(self.window, self.OUTLINE_CLR, top_left, bottom_left, self.LINE_WIDTH) #left
        pygame.draw.line(self.window, self.OUTLINE_CLR, bottom_left, bottom_right, self.LINE_WIDTH) #bottom

        #Draw the actual pieces
        pieces_to_draw = self.player.incoming_garbage
        next_x = self.GARBAGE_X_PX + self.GARBAGE_OFFSET_X
        next_y = self.GARBAGE_Y_PX + self.GARBAGE_HEIGHT_PX - self.GRIDSIZE
        
        next_y -= int(self.GRIDSIZE/5) #Give an extra boost
        for i in range(len(pieces_to_draw)):
            if next_y < self.GARBAGE_Y_PX:
                return

            clr = PlayerRenderer.garbage_clr(pieces_to_draw[i])
            self.draw_garbage_piece(0,0,next_x,next_y,self.GRIDSIZE,clr)
            next_y -= self.GRIDSIZE
            
            if i != len(pieces_to_draw) - 1 and pieces_to_draw[i] != pieces_to_draw[i+1]:
                next_y -= int(self.GRIDSIZE/5)

    @staticmethod
    def garbage_clr(num):
        '''
        Takes a num, returns a clr according to some rules decided based off how high it is.
        Desgined for garbage pieces.
        '''
        if num < 3:
            return Renderer.RED
        elif num < 8:
            return Renderer.ORANGE
        elif num < 14:
            return Renderer.YELLOW
        else:
            return Renderer.GREY

#Didn't write this function it is from: https://www.pygame.org/wiki/GradientCode
def fill_gradient(surface, color, gradient, rect=None, vertical=True, forward=True):
    """fill a surface with a gradient pattern
    Parameters:
    color -> starting color
    gradient -> final color
    rect -> area to fill; default is surface's rect
    vertical -> True=vertical; False=horizontal
    forward -> True=forward; False=reverse
    
    Pygame recipe: http://www.pygame.org/wiki/GradientCode
    """
    if rect is None: rect = surface.get_rect()
    x1,x2 = rect.left, rect.right
    y1,y2 = rect.top, rect.bottom
    if vertical: h = y2-y1
    else:        h = x2-x1
    if forward: a, b = color, gradient
    else:       b, a = color, gradient
    rate = (
        float(b[0]-a[0])/h,
        float(b[1]-a[1])/h,
        float(b[2]-a[2])/h
    )
    fn_line = pygame.draw.line
    if vertical:
        for line in range(y1,y2):
            color = (
                min(max(a[0]+(rate[0]*(line-y1)),0),255),
                min(max(a[1]+(rate[1]*(line-y1)),0),255),
                min(max(a[2]+(rate[2]*(line-y1)),0),255)
            )
            fn_line(surface, color, (x1,line), (x2,line))
    else:
        for col in range(x1,x2):
            color = (
                min(max(a[0]+(rate[0]*(col-x1)),0),255),
                min(max(a[1]+(rate[1]*(col-x1)),0),255),
                min(max(a[2]+(rate[2]*(col-x1)),0),255)
            )
            fn_line(surface, color, (col,y1), (col,y2))

    
