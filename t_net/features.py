# Store all features you can come up with here
import numpy as np


def number_of_holes(board):
    # Number of holes in the board (empty square with at least one block above it
    holes = 0
    for col in zip(*board):
        i = 0
        # find the first non-empty cell in a column
        while i < board.shape[0] and col[i] == 0:
            i += 1
        # find all the empty cells after the first non-empty cell.
        # These count as holes.
        holes += len([x for x in col[i+1:] if x == 0])

    return holes


def bumpiness(board):
    # Sum of the differences of heights between pair of columns
    total_bumpiness = 0
    max_bumpiness = 0
    min_ys = []
    # Find the first cell in each column that is non-zero.
    # This is the height of the column.
    for col in zip(*board):
        i = 0
        while i < board.shape[0] and col[i] == 0:
            i += 1
        min_ys.append(i)
    # Find the difference between consecutive heights.
    # This is the bumpiness.
    for i in range(len(min_ys) - 1):
        bumpiness = abs(min_ys[i] - min_ys[i+1])
        max_bumpiness = max(bumpiness, max_bumpiness)
        total_bumpiness += abs(min_ys[i] - min_ys[i+1])

    return total_bumpiness, max_bumpiness


def height(board):
    # Sum and maximum height of the board
    sum_height = 0
    max_height = 0
    min_height = board.shape[0]
    for col in zip(*board):
        i = 0
        # Find the height of the first column
        while i < board.shape[0] and col[i] == 0:
            i += 1
        # Update sum of heights, the max height, and the min height
        height = board.shape[0] - i
        sum_height += height
        if height > max_height:
            max_height = height
        elif height < min_height:
            min_height = height

    return sum_height, max_height, min_height


def calculate_linear_features(state):
    player = state.players[0]
    # Creates a vector of features to represent a player's board.
    num_rows = player.board.shape[0]
    num_cols = player.board.shape[1]
    # Extract the board
    board = player.board[5:num_rows-3, 3:num_cols-3]
    # calculate lines cleared, holes, bumpiness, and heights
    lines = player.num_lines_cleared
    holes = number_of_holes(board)
    total_bumpiness, max_bumpiness = bumpiness(board)
    sum_height, max_height, min_height = height(board)

    return np.array([lines, holes, total_bumpiness, sum_height])


def fetch_board(state):
    # Creates a vector of features to represent a player's board.
    num_rows = state.players[0].board.shape[0]
    num_cols = state.players[0].board.shape[1]
    # Extract the board
    board = state.players[0].board[5:num_rows-3, 3:num_cols-3].astype(bool).astype(int)
    board = board.reshape(1, 20, 10)
    return board


def calculate_mixed_cnn_features(state):
    # get a board and add one more dimension to it
    board = fetch_board(state).reshape(1, 20, 10, 1)
    # create an empty layer to hold a single value - number of cleared lines
    empty_layer = np.zeros((1, 20, 10, 1))
    # fill this value
    lines = state.players[0].num_lines_cleared
    empty_layer[0, 0, 0, 0] = lines
    # append the board with line layer
    board = np.append(board, empty_layer, axis=3)

    return board


def calculate_mixed_fc_features(state):
    # creates an array that consists of game board flattened and the number of lines cleared
    player = state.players[0]
    num_rows = state.players[0].board.shape[0]
    num_cols = state.players[0].board.shape[1]
    # Extract the board
    board = state.players[0].board[5:num_rows-3, 3:num_cols-3].astype(bool).astype(int)
    # flatten the board
    board = board.flatten()
    # append the array with the number of lines cleared
    lines = player.num_lines_cleared
    board = np.append(board, lines)
    return board

#I copied this in from multiplayer, figure it would be necessary soon: -Ian

def calculate_linear_features_multiplayer(state, player_id):
    
    def calculate_linear_features_for_player(state, player_id):

        player = state.players[player_id]
        # Creates a vector of features to represent a player's board.
        num_rows = player.board.shape[0]
        num_cols = player.board.shape[1]
        # Extract the board
        board = player.board[5:num_rows-3, 3:num_cols-3]
        # calculate lines cleared, holes, bumpiness, and heights
        lines = player.num_lines_cleared
        holes = number_of_holes(board)
        total_bumpiness, max_bumpiness = bumpiness(board)
        sum_height, max_height, min_height = height(board)

        features = np.array([lines, holes, total_bumpiness, sum_height])
        return features

    player_features = calculate_linear_features_for_player(state, player_id)
    opponent_features = []
    for opponent_id in range(len(state.players)):
        if opponent_id != player_id:
            opponent_features.append(calculate_linear_features_for_player(state, opponent_id))
    opponent_features = np.concatenate(opponent_features)

    return player_features, opponent_features