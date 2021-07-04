"""
Author:         Nate Koike & Zhaosen Guo
Date:           2021/3/15
Description:    Implement smart agents to play Othello, and experiment their
                efficiencies.
"""


from othello import *
import hw2
import numpy as np
import random, sys, copy, time


class MoveNotAvailableError(Exception):
    """Raised when a move isn't available."""
    pass


class OthelloTimeOut(Exception):
    """Raised when a player times out."""
    pass


class OthelloPlayer():
    """Parent class for Othello players."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass


class RandomPlayer(OthelloPlayer):
    """Plays a random move."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return random.choice(state.available_moves())


class HumanPlayer(OthelloPlayer):
    """Allows a human to play the game"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        available = state.available_moves()
        print("----- {}'s turn -----".format(state.current))
        print("Remaining time: {:0.2f}".format(remaining_time))
        print("Available moves are: ", available)
        move_string = input("Enter your move as 'r c': ")

        # Takes care of errant inputs and bad moves
        try:
            moveR, moveC = move_string.split(" ")
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_string, state.current))
            return self.make_move(state, remaining_time)


class MinimaxPlayer(OthelloPlayer):
    """ An basic AI player that uses the Minimax algorithm, a fixed
    search depth, and a simple heuristic evaluation function"""

    def __init__(self, color, heuristic):
        super(MinimaxPlayer, self).__init__(color)

        # The heuristic for the minimax algorithm
        self.heuristic = heuristic

    def minimax(self, state, max_turn, depth=1):
        """ A recursive function that performs minimax algorithm. """
        # Get all the available moves in the current state
        moves = state.available_moves()

        # If we are at a terminal node or the maximum depth
        if depth < 1 or state.game_over():
            return self.heuristic(state, self.color), None # no move

        best_move = None

        if max_turn:
            # The value of the position in node which we will calculate
            value = -np.inf

            # For every available move
            for move in moves:
                new_state = copy.deepcopy(state).apply_move(move)

                # Recursively run the minimax algorithm
                move_value, _ = self.minimax(new_state, not max_turn, depth - 1)

                # check to see
                if move_value > value:
                    value = move_value
                    best_move = move

            return value, best_move

        # it’s MIN’s turn
        else:
            # the value of the position in node which we will calculate
            value = np.inf

            # For every available move
            for move in moves:
                new_state = copy.deepcopy(state).apply_move(move)
                # Recursively run the minimax algorithm
                move_value, _ = self.minimax(new_state, not max_turn, depth - 1)

                # check to see
                if move_value < value:
                    value = move_value
                    best_move = move

            return value, best_move

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        _, move = self.minimax(state, True, 4)

        return move


class AlphaBetaPruner(OthelloPlayer):
    """ An AI player using alpha-beta pruning alogrithm, """

    def __init__(self, color, heuristic):
        super(AlphaBetaPruner, self).__init__(color)

        # The heuristic for the minimax algorithm
        self.heuristic = heuristic

    def alphabeta(self, state, a, b, max_turn, depth=1):
        """ The standard Alpha-beta pruning algorithm """
        moves = state.available_moves()

        # If there are no available moves (i.e. terminal node)
        if state.game_over() or depth < 1:
            return (self.heuristic(state, self.color), None)
        #
        # if state.game_over():
        #     if state.count(self.color) > state.count(opposite_color(self.color)):
        #         return (np.inf, None)
        #     return (-np.inf, None)

        # If it's MAX's turn
        if max_turn:
            value = -np.inf

            # Track the best move so far
            best = None

            # For every child state
            for move in moves:
                # Get the child state by applying the move
                child = copy.deepcopy(state).apply_move(move)

                # Find the maximum value of the current value and the child's
                # heuristic evaluation
                value = max(
                    value,
                    self.alphabeta(child, a, b, not max_turn, depth - 1)[0])

                # Set alpha to be the max of the value and the current alpha
                if value > a:
                    a = value
                    best = move

                # If alpha is greater than beta we can return
                if a >= b:
                    return (value, best)

            return (value, best)

        # It's MIN's turn
        else:
            value = np.inf

            # For every child
            for move in moves:

                # This is the same as MAX's turn but we're minimizing and using beta
                child = copy.deepcopy(state).apply_move(move)
                value = min(
                    value,
                    self.alphabeta(child, a, b, not max_turn, depth - 1)[0])

                if value < b:
                    b = value
                    best = move

                if a >= b:
                    return (value, move)

            return (value, move)

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        (_, move) = self.alphabeta(state, -np.inf, np.inf, True, 4)

        return move


class T1Player(OthelloPlayer):
    """ An AI player for testing, combining AB and better heuristics. """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        (_, move) = self.search(state, -np.inf, np.inf, True, 4)

        return move

    def search(self, state, a, b, max_turn, depth=1):
        """ The standard Alpha-beta pruning algorithm """
        moves = state.available_moves()

        # If there are no available moves (i.e. terminal node)
        if state.game_over() or depth < 1:
            return (self.heuristic(state, self.color), None)
        # if state.game_over():
        #     if state.count(self.color) > state.count(opposite_color(self.color)):
        #         return (np.inf, None)
        #     return (-np.inf, None)

        best = None
        if max_turn:
            value = -np.inf
            for move in moves:
                child = copy.deepcopy(state).apply_move(move)
                value = max(
                    value,
                    self.search(child, a, b, not max_turn, depth - 1)[0])
                if value > a:
                    a = value
                    best = move
                if a >= b:
                    return (value, best)
            return (value, best)
        else:
            value = np.inf
            for move in moves:
                child = copy.deepcopy(state).apply_move(move)
                value = min(
                    value,
                    self.search(child, a, b, not max_turn, depth - 1)[0])
                if value < b:
                    b = value
                    best = move
                if a >= b:
                    return (value, move)
            return (value, move)

    def heuristic(self, state, current_player):
        """ Given the state of Othello, return an evalution. """
        #### For easier access of player colors
        plrX = current_player # for Ma"X"
        plrN = opposite_color(current_player) # for Mi"N"

        #### Difference between players' pieces
        captures = self.normalizer(state.count(plrX), state.count(plrN))

        #### Compare players' mobility and stability
        mobility, stability = self.mobiliy_and_stability_check(state, plrX, plrN)

        #### Compare the corner patterns
        corner = self.corner_check(state, plrX, plrN)

        # Multipliers/weights for different elements
        # Inspired by Paul S. Rosenbloom
        total_moves = 64 - sum([row.count("empty") for row in state.board])

        cap_w = 30
        m_w = (2 * total_moves) if total_moves < 30 else (total_moves)
        cor_w = 7 * total_moves
        s_w = 36

        return cap_w * captures + m_w * mobility + cor_w * corner + s_w * stability


    def normalizer(self, max_value, min_value):
        """ Normalizes values calculated from heuristics for better weighing."""

        if (max_value + min_value) != 0:
            return 100 * (max_value - min_value) / (max_value + min_value)
        return 0

    def corner_check(self, state, plrX, plrN):
        """ Returns a normalized value for corner evalution of the board. """

        # Tracking the scores using player name as key
        scores = {plrX : 0, plrN : 0 }

        # Each list in the list represent 1 corner combination of 4 squares -
        # [0]: actual corner, [1]: X squares, [2/3]: C squares
        corners = [[state.board[0][0],state.board[1][1],state.board[0][1],state.board[1][0]],
        [state.board[7][0],state.board[6][1],state.board[7][1],state.board[6][0]],
        [state.board[7][7],state.board[6][6],state.board[7][6],state.board[6][7]],
        [state.board[0][7],state.board[1][6],state.board[0][6],state.board[1][7]]]

        # Check patterns for each corner
        for c in corners:
            if c[0] == "empty":
                for i in range(1,4):

                    # Punish placing around an uncaptured corner
                    if c[i] != "empty":
                        scores[opposite_color(c[i])] += 3
            else: # If the corner is captued:
                # Award capturing all 4 squares at the corner
                if len(set(c)) == 1:
                    scores[c[0]] += 15
                else:
                    c_color = c[0]
                    # Award capturing the corner
                    scores[c_color] += 4
                    for i in range(1,4):
                        # Punish placing around a captured opponent corner
                        if c[i] != c_color and c[i] != "empty":
                            scores[c_color] += 2
                        # Award placing around a captured self corner
                        if c[i] == c_color and c[i] != "empty":
                            scores[c_color] += 3

        return normalizer(scores[plrX], scores[plrN])

    def mobiliy_and_stability_check(self, state, plrX, plrN):
        """ Check the mobility and stability of the players in a given state. """

        #### Mobility
        ## Max's moves
        moveX = 0
        movelstX = []
        for r in range(8):
            for c in range(8):
                # Can only play in an empty square
                if state.board[r][c] == 'empty':
                    legal_move = False # We'll set to true to break out of loops when necessary
                    # We'll check in each direction for row and column
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr==0 and dc==0:
                                continue
                            if state.flanking(r + dr, c + dc, dr, dc, plrN, plrX):
                                moveX += 1
                                movelstX.append((r, c))
                                legal_move = True
                                break
                        if legal_move:
                            break
        ## Min's moves
        moveN = 0
        movelstN = []
        for r in range(8):
            for c in range(8):
                # Can only play in an empty square
                if state.board[r][c] == "empty":
                    legal_move = False # We'll set to true to break out of loops when necessary
                    # We'll check in each direction for row and column
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr==0 and dc==0:
                                continue
                            if state.flanking(r + dr, c + dc, dr, dc, plrX, plrN):
                                moveN += 1
                                movelstN.append((r, c))
                                legal_move = True
                                break
                        if legal_move:
                            break
        mobi_score = self.normalizer(moveX, moveN)

        #### Stability
        ## Check stability for X
        X_state = copy.deepcopy(state.board)

        for move in movelstN:
            # For each placement of N check which direction can it potentially flip
            r, c = move
            directions = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr==0 and dc==0:
                        continue
                    if state.flanking(r + dr, c + dc, dr, dc, plrX, plrN):
                        directions.append((dr, dc))

            # For each found directions mark all the ones that are flippable
            for dr, dc in directions:
                while r + dr >= 0 and c + dc >= 0 and r + dr < 8 and c + dc < 8 and X_state[r + dr][c + dc] == plrX:
                    X_state[r + dr][c + dc] = "weak"

        stableX = sum([row.count(plrX) for row in X_state])
        unstableX = sum([row.count("weak") for row in X_state])

        ## Check stability for N
        N_state = copy.deepcopy(state.board)

        for move in movelstX:
            # For each placement of N check which direction can it potentially flip
            r, c = move
            directions = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr==0 and dc==0:
                        continue
                    if state.flanking(r + dr, c + dc, dr, dc, plrN, plrX):
                        directions.append((dr, dc))

            # For each found directions mark all the ones that are flippable
            for dr, dc in directions:
                while r + dr >= 0 and c + dc >= 0 and r + dr < 8 and c + dc < 8 and N_state[r + dr][c + dc] == plrN:
                    N_state[r + dr][c + dc] = "wack"

        stableN = sum([row.count(plrN) for row in N_state])
        unstableN = sum([row.count("wack") for row in N_state])



        stab_score = self.normalizer(stableX + unstableN, stableN + unstableX)

        return mobi_score, stab_score


class TournamentPlayer(OthelloPlayer):
    """You should implement this class as your entry into the AI Othello tournament.
    You should implement other OthelloPlayers to try things out during your
    experimentation, but this is the only one that will be tested against your
    classmates' players."""


def simple_heuristic(state, color):
    """ A simple heuristic to determine the best move for the current player
        with ShortTermMaximizer. """

    return state.count(color)


def advanced_heuristic(state, color):
    """ An advanced heuristic that takes multiple factors into consideration."""
    #### For easier access of player colors
    plrX = color # for Ma"X"
    plrN = opposite_color(color) # for Mi"N"

    #### Difference between players' piece
    count = normalizer(state.count(plrX), state.count(plrN))

    #### Difference between corners captured
    ## Actual captured corners
    list_corners = [state.board[0][0], state.board[7][7],state.board[0][7],state.board[7][0]]
    actual_corners = normalizer(list_corners.count(plrX), list_corners.count(plrN))

    ## Pieces that can lead opponent to capture of the corners (all adjacents)
    list_adj = [state.board[1][1], state.board[6][1], # Diagonally adjacent ...
    state.board[1][6], state.board[6][6],
    state.board[0][1], state.board[0][6], # UDRL next to ...
    state.board[1][0], state.board[6][0], state.board[1][7],
    state.board[6][7], state.board[7][1],state.board[7][1]]

    # Switching plrX and plrN because it is an negative effect
    potential_corners = normalizer(list_adj.count(plrN), list_adj.count(plrX))
    # Put weights to the previously calculated values
    corners = (3 * actual_corners + potential_corners) / 4

    #### Difference between players' available_moves
    ## Max's moves
    moveX = 0
    for r in range(8):
        for c in range(8):
            # Can only play in an empty square
            if state.board[r][c] == 'empty':
                legal_move = False # We'll set to true to break out of loops when necessary
                # We'll check in each direction for row and column
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr==0 and dc==0:
                            continue
                        if state.flanking(r + dr, c + dc, dr, dc, plrN, plrX):
                            moveX += 1
                            legal_move = True
                            break
                    if legal_move:
                        break
    ## Min's moves
    moveN = 0
    for r in range(8):
        for c in range(8):
            # Can only play in an empty square
            if state.board[r][c] == 'empty':
                legal_move = False # We'll set to true to break out of loops when necessary
                # We'll check in each direction for row and column
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr==0 and dc==0:
                            continue
                        if state.flanking(r + dr, c + dc, dr, dc, plrX, plrN):
                            moveN += 1
                            legal_move = True
                            break
                    if legal_move:
                        break
    moves = normalizer(moveX, moveN)

    return count + corners + moves


def static_heuristic(state, color):
    """ A static heuristic function that lables each part of the board
        with a constant value. Proven comparable to simple_heuristic, but
        not better than others since it cannot dynamically evaluate. """

    plrX = color
    plrN = opposite_color(color)

    score_board = [
    [4, -3, 2, 2, 2, 2, -3, 4],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [2, -1, 1, 0, 0, 1, -1, 2],
    [2, -1, 0, 1, 1, 0, -1, 2],
    [2, -1, 0, 1, 1, 0, -1, 2],
    [2, -1, 1, 0, 0, 1, -1, 2],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [4, -3, 2, 2, 2, 2, -3, 4],
    ]

    plrX_score = 0
    plrN_score = 0

    for r in range(8):
        for c in range(8):
            place = state.board[r][c]
            if place == plrX:
                plrX_score += 1
            if place == plrN:
                plrN_score += 1

    return plrX_score - plrN_score


def normalizer(max_value, min_value):
    """ Normalizes values calculated from heuristics for better weighing."""

    if (max_value + min_value) != 0:
        return 100 * (max_value - min_value) / (max_value + min_value)
    return 0


def end_game_heuristic(state, color):
    """ An attempt to improve from the advanced_heuristic. """
    #### For easier access of player colors
    plrX = color # for Ma"X"
    plrN = opposite_color(color) # for Mi"N"

    #### Difference between players' piece
    captures = normalizer(state.count(plrX), state.count(plrN))

    #### Difference between players' mobilities
    ## Max's moves
    moveX = 0
    for r in range(8):
        for c in range(8):
            # Can only play in an empty square
            if state.board[r][c] == 'empty':
                legal_move = False # We'll set to true to break out of loops when necessary
                # We'll check in each direction for row and column
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr==0 and dc==0:
                            continue
                        if state.flanking(r + dr, c + dc, dr, dc, plrN, plrX):
                            moveX += 1
                            legal_move = True
                            break
                    if legal_move:
                        break
    ## Min's moves
    moveN = 0
    for r in range(8):
        for c in range(8):
            # Can only play in an empty square
            if state.board[r][c] == 'empty':
                legal_move = False # We'll set to true to break out of loops when necessary
                # We'll check in each direction for row and column
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr==0 and dc==0:
                            continue
                        if state.flanking(r + dr, c + dc, dr, dc, plrX, plrN):
                            moveN += 1
                            legal_move = True
                            break
                    if legal_move:
                        break
    mobility = normalizer(moveX, moveN)


    #### Compare the corner patterns
    corner  = corner_check(state, plrX, plrN)

    #### Compare stability between the players

    total_moves = 64 - sum([row.count("empty") for row in state.board])
    mobi_multi = 2 * total_moves if total_moves < 30 else total_moves

    return 30 * captures + mobi_multi * mobility + 6 * total_moves * corner


def corner_check(state, plrX, plrN):
    """ Returns a normalized value for corner evalution of the board. """

    # Tracking the scores using player name as key
    scores = {plrX : 0, plrN : 0 }

    # Each list in the list represent 1 corner combination of 4 squares -
    # [0]: actual corner, [1]: X squares, [2/3]: C squares
    corners = [[state.board[0][0],state.board[1][1],state.board[0][1],state.board[1][0]],
    [state.board[7][0],state.board[6][1],state.board[7][1],state.board[6][0]],
    [state.board[7][7],state.board[6][6],state.board[7][6],state.board[6][7]],
    [state.board[0][7],state.board[1][6],state.board[0][6],state.board[1][7]]]

    # Check patterns for each corner
    for c in corners:
        if c[0] == "empty":
            for i in range(1,4):

                # Punish placing around an uncaptured corner
                if c[i] != "empty":
                    scores[opposite_color(c[i])] += 3
        else: # If the corner is captued:
            # Award capturing all 4 squares at the corner
            if len(set(c)) == 1:
                scores[c[0]] += 15
            else:
                c_color = c[0]
                # Award capturing the corner
                scores[c_color] += 4
                for i in range(1,4):
                    # Punish placing around a captured opponent corner
                    if c[i] != c_color and c[i] != "empty":
                        scores[c_color] += 2
                    # Award placing around a captured self corner
                    if c[i] == c_color and c[i] != "empty":
                        scores[c_color] += 3

    return normalizer(scores[plrX], scores[plrN])


def helmuth_heuristic(state):
    """ THe heuristic used by Professor T. Helmuth, used for debug only. """
    return state.count(state.current) - state.count(opposite_color(state.current))

################################################################################

def main():
    """Plays the game."""

    # black_player = AlphaBetaPruner("black", end_game_heuristic)
    # white_player = T1Player("white")
    #
    # game = OthelloGame(black_player, white_player, verbose=False)
    #
    # winner = game.play_game_data()
    #
    # print("black (X): {}, time remains: {}".format(
    #     game.board.count('black'), game.black_time))
    # print("white (O): {}, time remains: {}".format(
    #     game.board.count('white'), game.white_time))
    # print(game.board)
    #
    # if not game.verbose:
    #     print("Winner is", winner)
    #
    # print(0)
    # print(0)
    # print(0)

    black_player = T1Player("black")
    white_player = AlphaBetaPruner("white", end_game_heuristic)

    game = OthelloGame(black_player, white_player, verbose=False)

    winner = game.play_game_data()

    print("black (X): {}, time remains: {}".format(
        game.board.count('black'), game.black_time))
    print("white (O): {}, time remains: {}".format(
        game.board.count('white'), game.white_time))
    print(game.board)

    if not game.verbose:
        print("Winner is", winner)


if __name__ == "__main__":
    main()
