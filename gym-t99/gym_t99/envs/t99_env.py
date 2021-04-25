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

        # step 1: process all events
        for event in self.state.event_queue:
            self._process(event)
        # step 2: process all actions
        for i in range(self.state.players):
            # if this is the first player, controlled by AI
            if i==0:
                # use action passed in command option of step
                self._apply_action(i, action)
            # if this player is controlled by environment AI
            else:
                # first, generate action
                action = enemy.action(self.state.observe(i))
                # and then apply it
                self._apply_action(i, action)
        # step 3: update all player's with in-game mechanics
        for i in range(self.state.players):
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
                temp_board = self._apply(self.state.players[i].board, self.state.players[i].piece_current)
                # append the list with their board
                frame.append(temp_board)
            
            if self.pygame_started: #Will let us switch between debug or human mode in same session if desired
                self.renderer.quit()
                self.pygame_started = False

        elif mode == "human":
            # TODO:  Ian's code here
            if not self.pygame_started:
                height , width = self.state.players[0].board.shape #Get dimensions of board
                self.renderer = Renderer(height,width,show_window=show_window)
                self.pygame_started = True
            
            self.renderer.draw_screen(self.state)
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


#TODO: 
    #Create a class in the this class, Renderer that handles all the utility of creating the image
    #The only constant it will use is the size of a side of the block
        #Later this can be made into some sort of step function with respect to the screen size
    #Using this block size can then half it or quarter it for other non-primary boards

class Renderer():

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


    def __init__(self,board_height,board_width,show_window=False,gridsize=30):
        self.GRIDSIZE = gridsize
        self.BOARD_HEIGHT = board_height 
        self.BOARD_WIDTH = board_width
        self.BOARD_HEIGHT_PX = self.BOARD_HEIGHT * self.GRIDSIZE
        self.BOARD_WIDTH_PX = self.BOARD_WIDTH * self.GRIDSIZE

        #Headless means the window will not be shown
        #The window does not work with my Jupyter notebook
        
        if not show_window:
            environ["SDL_VIDEODRIVER"] = "dummy" #Makes window not appear
        
        pygame.init()
        self.window = pygame.display.set_mode((self.BOARD_WIDTH_PX , self.BOARD_HEIGHT_PX))

    def draw_screen(self,state):
        self.window.fill(Renderer.BLACK)
        self.draw_grid()
        self.draw_all_blocks(state.players[0].board)
        pygame.display.update()
        
    def draw_grid(self):
        '''
        Draws the grid on the screen for the pieces
        '''
        for i in range(self.BOARD_WIDTH+1):
            pygame.draw.line(self.window, Renderer.GREY, (i * self.GRIDSIZE, 0), (i * self.GRIDSIZE, self.BOARD_HEIGHT_PX), 1)

        for i in range(self.BOARD_HEIGHT+1):
            pygame.draw.line(self.window, Renderer.GREY, (0, i * self.GRIDSIZE), (self.BOARD_WIDTH_PX, i * self.GRIDSIZE), 1)

    def draw_all_blocks(self,board):
        for row in range(0,board.shape[0]):
            for col in range(0,board.shape[1]):
                self.draw_block(col,row,board[row][col])


    def draw_block(self,grid_x,grid_y,piece_int):
        '''
        Takes a grid_x (column) and a grid_y (row), and the int of the piece
        Draws the piece ont the window
        '''
        if piece_int == 0: #Skip empty squares
            return 
        x_px = grid_x * self.GRIDSIZE
        y_px = grid_y * self.GRIDSIZE
        clr = Renderer.piece_colors[piece_int]
        pygame.draw.rect(self.window, clr, (x_px, y_px, self.GRIDSIZE-2, self.GRIDSIZE-2), 0)
        
        if piece_int != 10: #Add highlight around all non-white pieces
            print("didd it")
            pygame.draw.rect(self.window, Renderer.WHITE, (x_px, y_px, self.GRIDSIZE, self.GRIDSIZE), 1) #Outline
    
    def save_screen_as_image(self):
        pygame.image.save(self.window, "screenshot.png")

    def quit(self):
        '''
        Quits pygame
        Not currently working for some reason
        '''
        pygame.display.quit()
        pygame.quit()

