import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .state import *

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


    def render(self, mode='human',show_window=False):
        """
        :param str mode: mode in which rendering works. If debug, returns numpy matrices for each player
        :param bool show_window: assuming mode='human', says whether to show the window or not
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
            # TODO:  Ian's code here
            if not self.pygame_started:
                self.renderer = Renderer(self.state.players,show_window=show_window)
                self.pygame_started = True
            
            self.renderer.draw_screen()
            self.renderer.save_screen_as_image()
            
            frame=None

        return frame


    def close(self):
        if self.pygame_started:
            self.renderer.quit()
            
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
        self.SMALL_PADDING_PX, self.MEDIUM_PADDING_PX, self.LARGE_PADDING_PX = 2 * self.SECONDARY_BOARD_GRIDSIZE, 4 * self.SECONDARY_BOARD_GRIDSIZE, self.MAIN_BOARD_GRIDSIZE
        self.VERT_PADDING_PX, self.SIDE_PADDING_PX = self.LARGE_PADDING_PX, self.SMALL_PADDING_PX

        #Set widths of important areas
        self.NPC_WIDTH_PX = 7 * self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_WIDTH + 6 * self.SMALL_PADDING_PX #Width of single side of NPCS
        self.NPC_HEIGHT_PX = 7 * self.SECONDARY_BOARD_GRIDSIZE * self.BOARD_HEIGHT + 6 * self.MEDIUM_PADDING_PX #Width of single side of NPCS
        
        self.PLAYER_WIDTH_PX = self.MAIN_BOARD_GRIDSIZE * self.BOARD_WIDTH #TODO: do this in a more programmtic way in the future, when player has its own rendering class
        
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
        x_orig = self.SIDE_PADDING_PX + self.NPC_WIDTH_PX + self.MEDIUM_PADDING_PX
        y_orig = self.VERT_PADDING_PX
        self.player_board = PlayerRenderer(self.player,x_orig,y_orig,self.MAIN_BOARD_GRIDSIZE,self.window, self.top_rows_ignore, self.num_walls_side, self.num_walls_bottom)


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
        self.window.fill(Renderer.BLACK)
        for board_renderer in self.npc_board_renderers:
            board_renderer.draw_board()
        
        self.player_board.draw_board()
        #TODO: Add section to draw main player's board
        pygame.display.update()

    
    def save_screen_as_image(self):
        '''
        Saves the screen as an image, will update this in the future to meet needs of learning
        '''
        pygame.image.save(self.window, "screenshot.png")

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
        TODO : Handle if boards player is even alive!!
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
            self.LINE_WIDTH = max(1,int(gridsize // 30))
            self.ORIGIN_X_PX = origin_x_px
            self.ORIGIN_Y_PX = origin_y_px
            
            self.fontsize = int(self.GRIDSIZE*4)
            self.arial = pygame.font.SysFont('Arial Black', self.fontsize)


        def draw_board(self):
            '''
            Draws a board
            '''
        
            self.draw_outline()
            if self.player.place != None:
                self.draw_elimination()
            else:
                self.draw_all_blocks()
        
        def draw_elimination(self):
            string1 = "K.O"
            if len(str(self.player.place)) == 1:
                string2 = "  " + str(self.player.place)
            else:
                string2 = " " + str(self.player.place)
            place_text = self.arial.render(string1, True, Renderer.WHITE)
            place_text2 = self.arial.render(string2, True, Renderer.WHITE)
            x1 = int(self.BOARD_WIDTH_PX/8) + self.ORIGIN_X_PX
            y1 = int(self.BOARD_HEIGHT_PX/5) + self.ORIGIN_Y_PX
            x2 = x1
            y2 = y1 + self.fontsize
            self.window.blit(place_text,(x1,y1))
            self.window.blit(place_text2,(x2,y2))
        
        def draw_outline(self):
            top_left = (self.ORIGIN_X_PX - 1, self.ORIGIN_Y_PX - 1)
            top_right = (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX, self.ORIGIN_Y_PX-1)
            bottom_left = (self.ORIGIN_X_PX-1, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX )
            bottom_right = (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX)
            pygame.draw.line(self.window, Renderer.GREY, top_left, bottom_left, self.LINE_WIDTH)
            pygame.draw.line(self.window, Renderer.GREY, top_left, top_right, self.LINE_WIDTH)
            pygame.draw.line(self.window, Renderer.GREY, top_right, bottom_right, self.LINE_WIDTH)
            pygame.draw.line(self.window, Renderer.GREY, bottom_left, bottom_right, self.LINE_WIDTH)

        def draw_grid(self):
            '''
            Draws the background grid on the screen for the pieces
            '''
            for i in range(1,self.BOARD_WIDTH):
                pygame.draw.line(self.window, Renderer.GREY, (self.ORIGIN_X_PX + i * self.GRIDSIZE, self.ORIGIN_Y_PX), (self.ORIGIN_X_PX + i * self.GRIDSIZE, self.ORIGIN_Y_PX + self.BOARD_HEIGHT_PX), self.LINE_WIDTH)

            for i in range(1,self.BOARD_HEIGHT):
                pygame.draw.line(self.window, Renderer.GREY, (self.ORIGIN_X_PX, self.ORIGIN_Y_PX + i * self.GRIDSIZE), (self.ORIGIN_X_PX + self.BOARD_WIDTH_PX , self.ORIGIN_Y_PX + i * self.GRIDSIZE), self.LINE_WIDTH)

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
            #pygame.draw.rect(self.window, clr, (x_px, y_px, self.GRIDSIZE-(2*self.SHADOW_SIZE_PX), self.GRIDSIZE-(2*self.SHADOW_SIZE_PX)), 0)
            pygame.draw.rect(self.window, clr, (x_px, y_px, self.GRIDSIZE, self.GRIDSIZE), 0)
            
            #if piece_int != 10: #Add highlight around all non-grey pieces
            #    pygame.draw.rect(self.window, Renderer.WHITE, (x_px, y_px, self.GRIDSIZE, self.GRIDSIZE), self.SHADOW_SIZE_PX) #Outline

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
            
    def draw_board(self):
            '''
            Draws a board
            '''
            self.draw_outline()
            if self.player.place != None:
                self.draw_elimination()
            else:
                self.draw_grid()
                self.draw_all_blocks()
            
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
            pygame.draw.rect(self.window, clr, (x_px, y_px, self.GRIDSIZE-(2*self.SHADOW_SIZE_PX), self.GRIDSIZE-(2*self.SHADOW_SIZE_PX)), 0)
            
            if piece_int != 10: #Add highlight around all non-grey pieces
                pygame.draw.rect(self.window, Renderer.WHITE, (x_px, y_px, self.GRIDSIZE, self.GRIDSIZE), self.SHADOW_SIZE_PX) #Outline 
