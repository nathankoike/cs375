"""
Author:         Nate Koike & Zhaosen Guo
Date:           2021/3/15
Description:    Implement smart agents to play Othello, and experiment their
                efficiencies.

                Team name: (-b±√(b^2-4ac))/2a
"""


from othello import *
import hw2
import numpy as np
import random, sys, copy, time, math


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
                new_state = state.apply_move(move)

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
                new_state = state.apply_move(move)
                # Recursively run the minimax algorithm
                move_value, _ = self.minimax(new_state, not max_turn, depth - 1)

                # check to see
                if move_value < value:
                    value = move_value
                    best_move = move

            return value, best_move

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        return self.minimax(state, True, 4)[1]


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

        # Track the best move so far
        best = None

        # If it's MAX's turn
        if max_turn:
            value = -np.inf

            # For every child state
            for move in moves:
                # Get the child state by applying the move
                child = state.apply_move(move)

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
                child = state.apply_move(move)
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


class SearchTree:
    def __init__(self, wins, state, parent=None):
        # The number of wins reached through this state/move
        self.wins = wins

        # The state reached through the move
        self.state = state

        # This node's parent node
        self.parent = parent

        # Nodes will always initialize with 1 play and no children
        self.plays = 0
        self.children = []

    def update(self, win):
        """ Increment all the values and propagate these changes all the way
        back up the tree """
        # Add 1 to the number of times this node has been played
        self.plays += 1

        # Add the winning value {0: loss, 0.5: draw, 1: win}
        self.wins += win

        # If the parent is the root node
        if self.parent == None:
            return # We're done

        # Send the data back up the tree
        self.parent.update(win)

    def ucb(self):
        """ Calculate the upper confidence bound of the node """
        # If the move is new, pick it
        if self.plays < 1:
            return np.inf

        # The constant used in the UCB calculation, can be tuned
        constant = math.sqrt(2)

        # This is just an coded version of the formula from class
        return (
            (self.wins / self.plays) +
            (constant * math.sqrt(math.log(self.parent.plays / self.plays))))

    def fully_expanded(self):
        """ Determine if we've already seen a child before """
        # A node is fully expanded if the number of available moves is the same
        # as the number of children
        return len(self.state.available_moves()) == len(self.children)

    def traverse(self):
        """ Traverse the tree and return the leaf node with the best UCB """
        # Start the search at the current node
        node = self

        # While the node is fully expanded
        while node.fully_expanded():
            # Check for terminal node
            if node.children == []:
                return node # The game is over, so we can just return this leaf

            # Track the best child and the UCB value of the child
            best_child, best_ucb = None, -np.inf

            # For every child...
            for child in node.children:
                # Get the UCB of the child
                child_ucb = child.ucb()

                # If the current child is more promising than the current best
                if child_ucb > best_ucb:
                    best_child, best_ucb = child, child_ucb

            # Reset the node to be the best child
            node = best_child

        # We found a node that isn't fully expanded so get the next child
        move = node.state.available_moves()[len(node.children)]

        # Make a new node for the child
        node.children.append(SearchTree(0, node.state.apply_move(move), node))

        # Return the most recently added child
        return node.children[-1] # We have a new direction to explore


class MCTSPlayer(OthelloPlayer):
    def __init__(self, color, heuristic):
        super(MCTSPlayer, self).__init__(color)

        # The heuristic for the minimax algorithm
        self.heuristic = heuristic

    def sim_game(self, state):
        """ Randomly simulate a game from the given state, returning the winner
        of the random game """
        # If the game is over, return the outcome of the game
        if state.game_over():
            return state.winner()

        # Get a random, legal move in the game
        move = random.choice(state.available_moves())

        # Recursively pick random moves until the game ends
        return self.sim_game(state.apply_move(move))

    def mcts(self, state, move_time):
        """ Perform MCTS on the current state to find a decent-looking move """
        # The parent for the new nodes
        parent = None

        # A placeholder for the tree we will be searching later
        tree = SearchTree(0, state, None)

        # Check for the edge case that the game is over
        if state.game_over():
            return None

        # While we still have time
        while move_time - time.time() > 0:
            # Get the most interesting leaf
            node = tree.traverse()

            # Simulate the game using the most promising move found so far
            outcome = self.sim_game(node.state)

            # Get the win value
            if outcome == 'draw':
                win = .5
            elif outcome == self.color:
                win = 1
            else:
                win = 0

            # Update the values in the tree
            node.update(win)

        # Store the best node and its rating
        best_child, best_confidence = None, -np.inf

        # For every child
        for child in tree.children:
            # Discard children with no plays
            if child.plays < 1:
                continue

            # Find the average heuristic value of the node's children
            avg_h = 0

            for next_child in child.children:
                avg_h += self.heuristic(next_child.state, self.color)

            if len(child.children) > 0:
                avg_h /= len(child.children)
            else:
                avg_h = self.heuristic(child.state, self.color)

            # Get some measure of how well the child will perform
            child_confidence = (child.plays / math.sqrt(2)) + avg_h

            # If this one is better than the others
            if child_confidence > best_confidence:
                best_child, best_confidence = child, child_confidence

        # Return the best child
        return state.available_moves()[tree.children.index(best_child)]

    def make_move(self, state, remaining_time):
        """ Return the best move """
        return self.mcts(state, time.time() + (remaining_time / 10))


class T1Player(OthelloPlayer):
    """ An AI player for testing, combining AB and better heuristics. """

    def normalizer(self, max_value, min_value):
        """ Normalizes values calculated from heuristics for better weighing."""
        # As long as we have a nonzero sum
        if (max_value + min_value) != 0:
            # Normalize the values to some non-trivial amount
            return 100 * (max_value - min_value) / (max_value + min_value)

        # Zero is zero
        return 0

    def corner_check(self, state, plrX, plrN):
        """ Returns a normalized value for corner evalution of the board. """
        # Tracking the scores using player name as key
        scores = { plrX : 0, plrN : 0 }

        # Each list in the list represent 1 corner combination of 4 squares -
        # [0]: actual corner, [1]: diagonal squares, [2/3]: adjacent squares
        corners = [
            [ # The top left corner
                state.board[0][0],
                state.board[1][1],
                state.board[0][1],
                state.board[1][0]],
            [ # The bottom left corner
                state.board[7][0],
                state.board[6][1],
                state.board[7][1],
                state.board[6][0]],
            [ # The bottom right corner
                state.board[7][7],
                state.board[6][6],
                state.board[7][6],
                state.board[6][7]],
            [ # The top right corner
                state.board[0][7],
                state.board[1][6],
                state.board[0][6],
                state.board[1][7]]]

        # Check patterns for each corner
        for corner in corners:
            # If the corner is empty
            if corner[0] == "empty":
                # For indices [1, 3]
                for i in range(1,4):
                    # Punish placing around an uncaptured corner
                    if corner[i] != "empty":
                        scores[opposite_color(corner[i])] += 3

            # If the corner is captued:
            else:
                # Award capturing all 4 squares at the corner
                if len(set(corner)) == 1:
                    scores[corner[0]] += 15
                # Not all four corner squares are captured
                else:
                    corner_color = corner[0]
                    # Award capturing the corner
                    scores[corner_color] += 4
                    for i in range(1,4):
                        # Punish placing around a captured opponent corner
                        if corner[i] != corner_color and corner[i] != "empty":
                            scores[corner_color] += 2

                        # Award placing around a captured self corner
                        if corner[i] == corner_color and corner[i] != "empty":
                            scores[corner_color] += 3

        # Normalize the data and return it
        return normalizer(scores[plrX], scores[plrN])

    def mobility(self, max_moves, min_moves):
        """ Find the mobility score for the current board position """
        # The number of MAX and MIN moves
        moveX = len(max_moves)
        moveN = len(min_moves)

        # Return the mobility score of the position
        return self.normalizer(moveX, moveN)

    def stability(self, state, max_moves, min_moves, plrX, plrN):
        """ Find the stability score for the current board position """
        # The number of unstable pieces for MAX and MIN
        unstableX = 0
        unstableN = 0

        # For every move in MIN's possible moves
        for move in min_moves:
            (row, col) = move.pair

            # Directions in which MAX could lose control
            directions = []

            # For every direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # If there won't be any change
                    if dr==0 and dc==0:
                        continue

                    # If MAX is flanked by MIN in that direction
                    if state.flanking(row + dr, col + dc, dr, dc, plrX, plrN):
                        # Add the direction to the list
                        directions.append((dr, dc))

            # For each volatile direction...
            for dr, dc in directions:
                # Count all the pieces that are flippable
                while in_bounds(row + dr, col + dc) and \
                    state.board[row + dr][col + dc] == plrX:
                    unstableX += 1

        # For every move in MAX's possible moves
        for move in max_moves:
            (row, col) = move.pair

            # Directions in which MAX could lose control
            directions = []

            # For every direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # If there won't be any change
                    if dr==0 and dc==0:
                        continue

                    # If MIN is flanked by MAX in that direction
                    if state.flanking(row + dr, col + dc, dr, dc, plrN, plrX):
                        # Add the direction to the list
                        directions.append((dr, dc))

            # For each volatile direction...
            for dr, dc in directions:
                # Count all the pieces that are flippable
                while in_bounds(row + dr, col + dc) and \
                    state.board[row + dr][col + dc] == plrX:
                    unstableN += 1

        # The number of stable pieces for MAX and MIN
        stableX = state.count(plrX) - unstableX
        stableN = state.count(plrN) - unstableN

        # Return the stability score of the position
        return self.normalizer(stableX + unstableN, stableN + unstableX)

    def mobiliy_and_stability_check(self, state, plrX, plrN):
        """ Check the mobility and stability of the players in a given state """
        # Find all of MAX's moves
        max_moves = state.available_moves()

        # Create a deep copy of the state and force it to be MIN's turn
        min_state = copy.deepcopy(state)
        min_state.current = plrN

        # Find all of MIN's moves
        min_moves = min_state.available_moves()

        # Return the pair of mobility and stability scores
        return self.mobility(max_moves, min_moves), \
        self.stability(state, max_moves, min_moves, plrX, plrN)

    def heuristic(self, state, current_player):
        """ Given the state of Othello, return an evalution. """
        # For easier access of player colors
        plrX = current_player # for Ma"X"
        plrN = opposite_color(current_player) # for Mi"N"

        # Difference between players' pieces
        captures = self.normalizer(state.count(plrX), state.count(plrN))

        # Compare players' mobility and stability
        mobi, stabi = self.mobiliy_and_stability_check(state, plrX, plrN)

        # Compare the corner patterns
        corner = self.corner_check(state, plrX, plrN)

        # Multipliers/weights for different elements
        # Inspired by Paul S. Rosenbloom
        total_moves = 64 - sum([row.count("empty") for row in state.board])

        # Weights that we can tune for a better evaluation
        cap_w = 30
        m_w = (2 * total_moves) if total_moves < 30 else (total_moves)
        cor_w = 7 * total_moves
        s_w = 36

        # Apply the weights and return the sum of the factors discovered above
        return cap_w * captures + m_w * mobi + cor_w * corner + s_w * stabi

    def search(self, state, a, b, max_turn, depth=1):
        """ The standard Alpha-beta pruning algorithm """
        # Get all the available moves
        moves = state.available_moves()

        # If there are no available moves (i.e. terminal node) or we have hit
        # the maximum depth
        if moves == [] or depth < 1:
            return (self.heuristic(state, self.color), None)

        # Track the best move so far
        best = None

        if max_turn:
            # A very low number
            value = -np.inf

            # For every available move
            for move in moves:
                child = state.apply_move(move)

                # Get the largest heuristic value
                value = max(
                    value,
                    self.search(child, a, b, not max_turn, depth - 1)[0])

                # If the largest heuristic value is larger than alpha
                if value > a:
                    # Update alpha and the best move
                    a = value
                    best = move

                # If the best move is seemingly at least as good as what our
                # opponent can produce
                if a >= b:
                    return (value, best) # Return the value and the move

            # Return back the value and the best move we found
            return (value, best)

        # It's MIN's turn
        else:
            # A really high number
            value = np.inf

            # This is the same as MAX's turn but trying to find a small number
            # instead
            for move in moves:
                child = state.apply_move(move)

                value = min(
                    value,
                    self.search(child, a, b, not max_turn, depth - 1)[0])

                if value < b:
                    b = value
                    best = move

                if a >= b:
                    return (value, move)

            return (value, move)

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        return self.search(state, -np.inf, np.inf, True, 4)[1]


class T2Player(OthelloPlayer):
    """ An AI player for testing, combining AB and better heuristics. """

    def normalizer(self, max_value, min_value):
        """ Normalizes values calculated from heuristics for better weighing."""
        # As long as we have a nonzero sum
        if (max_value + min_value) != 0:
            # Normalize the values to some non-trivial amount
            return 100 * (max_value - min_value) / (max_value + min_value)

        # Zero is zero
        return 0

    def corner_check(self, state, plrX, plrN):
        """ Returns a normalized value for corner evalution of the board. """
        # Tracking the scores using player name as key
        scores = { plrX : 0, plrN : 0 }

        # Each list in the list represent 1 corner combination of 4 squares -
        # [0]: actual corner, [1]: diagonal squares, [2/3]: adjacent squares
        corners = [
            [ # The top left corner
                state.board[0][0],
                state.board[1][1],
                state.board[0][1],
                state.board[1][0]],
            [ # The bottom left corner
                state.board[7][0],
                state.board[6][1],
                state.board[7][1],
                state.board[6][0]],
            [ # The bottom right corner
                state.board[7][7],
                state.board[6][6],
                state.board[7][6],
                state.board[6][7]],
            [ # The top right corner
                state.board[0][7],
                state.board[1][6],
                state.board[0][6],
                state.board[1][7]]]

        # Check patterns for each corner
        for corner in corners:
            # If the corner is empty
            if corner[0] == "empty":
                # For indices [1, 3]
                for i in range(1,4):
                    # Punish placing around an uncaptured corner
                    if corner[i] != "empty":
                        scores[opposite_color(corner[i])] += 3

            # If the corner is captued:
            else:
                # Award capturing all 4 squares at the corner
                if len(set(corner)) == 1:
                    scores[corner[0]] += 15
                # Not all four corner squares are captured
                else:
                    corner_color = corner[0]
                    # Award capturing the corner
                    scores[corner_color] += 4
                    for i in range(1,4):
                        # Punish placing around a captured opponent corner
                        if corner[i] != corner_color and corner[i] != "empty":
                            scores[corner_color] += 2

                        # Award placing around a captured self corner
                        if corner[i] == corner_color and corner[i] != "empty":
                            scores[corner_color] += 3

        # Normalize the data and return it
        return normalizer(scores[plrX], scores[plrN])

    def mobility(self, max_moves, min_moves):
        """ Find the mobility score for the current board position """
        # The number of MAX and MIN moves
        moveX = len(max_moves)
        moveN = len(min_moves)

        # Return the mobility score of the position
        return self.normalizer(moveX, moveN)

    def stability(self, state, max_moves, min_moves, plrX, plrN):
        """ Find the stability score for the current board position """
        # The number of unstable pieces for MAX and MIN
        unstableX = 0
        unstableN = 0

        # For every move in MIN's possible moves
        for move in min_moves:
            (row, col) = move.pair

            # Directions in which MAX could lose control
            directions = []

            # For every direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # If there won't be any change
                    if dr==0 and dc==0:
                        continue

                    # If MAX is flanked by MIN in that direction
                    if state.flanking(row + dr, col + dc, dr, dc, plrX, plrN):
                        # Add the direction to the list
                        directions.append((dr, dc))

            # For each volatile direction...
            for dr, dc in directions:
                # Count all the pieces that are flippable
                while in_bounds(row + dr, col + dc) and \
                    state.board[row + dr][col + dc] == plrX:
                    unstableX += 1

        # For every move in MAX's possible moves
        for move in max_moves:
            (row, col) = move.pair

            # Directions in which MAX could lose control
            directions = []

            # For every direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # If there won't be any change
                    if dr==0 and dc==0:
                        continue

                    # If MIN is flanked by MAX in that direction
                    if state.flanking(row + dr, col + dc, dr, dc, plrN, plrX):
                        # Add the direction to the list
                        directions.append((dr, dc))

            # For each volatile direction...
            for dr, dc in directions:
                # Count all the pieces that are flippable
                while in_bounds(row + dr, col + dc) and \
                    state.board[row + dr][col + dc] == plrX:
                    unstableN += 1

        # The number of stable pieces for MAX and MIN
        stableX = state.count(plrX) - unstableX
        stableN = state.count(plrN) - unstableN

        # Return the stability score of the position
        return self.normalizer(stableX + unstableN, stableN + unstableX)

    def mobiliy_and_stability_check(self, state, plrX, plrN):
        """ Check the mobility and stability of the players in a given state """
        # Find all of MAX's moves
        max_moves = state.available_moves()

        # Create a deep copy of the state and force it to be MIN's turn
        min_state = copy.deepcopy(state)
        min_state.current = plrN

        # Find all of MIN's moves
        min_moves = min_state.available_moves()

        # Return the pair of mobility and stability scores
        return self.mobility(max_moves, min_moves), \
        self.stability(state, max_moves, min_moves, plrX, plrN)

    def heuristic(self, state, current_player):
        """ Given the state of Othello, return an evalution. """
        # For easier access of player colors
        plrX = current_player # for Ma"X"
        plrN = opposite_color(current_player) # for Mi"N"

        # Difference between players' pieces
        captures = self.normalizer(state.count(plrX), state.count(plrN))

        # Compare players' mobility and stability
        mobi, stabi = self.mobiliy_and_stability_check(state, plrX, plrN)

        # Compare the corner patterns
        corner = self.corner_check(state, plrX, plrN)

        # Multipliers/weights for different elements
        # Inspired by Paul S. Rosenbloom
        total_moves = 64 - sum([row.count("empty") for row in state.board])

        # Weights that we can tune for a better evaluation
        cap_w = 30
        m_w = (2 * total_moves) if total_moves < 30 else (total_moves)
        cor_w = 7 * total_moves
        s_w = 36

        # Apply the weights and return the sum of the factors discovered above
        return cap_w * captures + m_w * mobi + cor_w * corner + s_w * stabi

    def sim_game(self, state):
        """ Randomly simulate a game from the given state, returning the winner
        of the random game """
        # If the game is over, return the outcome of the game
        if state.game_over():
            return state.winner()

        # Get a random, legal move in the game
        move = random.choice(state.available_moves())

        # Recursively pick random moves until the game ends
        return self.sim_game(state.apply_move(move))

    def mcts(self, state, move_time):
        """ Perform MCTS on the current state to find a decent-looking move """
        # The parent for the new nodes
        parent = None

        # A placeholder for the tree we will be searching later
        tree = SearchTree(0, state, None)

        # Check for the edge case that the game is over
        if state.game_over():
            return None

        # While we still have time
        while move_time - time.time() > 0:
            # Get the most interesting leaf
            node = tree.traverse()

            # Simulate the game using the most promising move found so far
            outcome = self.sim_game(node.state)

            # Get the win value
            if outcome == 'draw':
                win = .5
            elif outcome == self.color:
                win = 1
            else:
                win = 0

            # Update the values in the tree
            node.update(win)

        # Store the best node and its rating
        best_child, best_confidence = None, -np.inf

        # For every child
        for child in tree.children:
            # Discard children with no plays
            if child.plays < 1:
                continue

            # Track the average heuristic value of the node's children
            avg_h = 0

            # For every child of the child (grandchild?)
            for next_child in child.children:
                # Divide the heuristic value of the grandchild by the
                # number of grandchildren and add it to the average total
                avg_h += self.heuristic(next_child.state, self.color) / \
                len(child.children)

            # Get some measure of how well the child will perform
            child_confidence = (child.plays / math.sqrt(2)) + avg_h

            # If this one is better than the others
            if child_confidence > best_confidence:
                best_child, best_confidence = child, child_confidence

        # Return the best child
        return state.available_moves()[tree.children.index(best_child)]

    def abp(self, state, a, b, max_turn, depth=1):
        """ The standard Alpha-beta pruning algorithm """
        # Get all the available moves
        moves = state.available_moves()

        # If there are no available moves (i.e. terminal node) or we have hit
        # the maximum depth
        if moves == [] or depth < 1:
            return (self.heuristic(state, self.color), None)

        # Track the best move so far
        best = None

        if max_turn:
            # A very low number
            value = -np.inf

            # For every available move
            for move in moves:
                child = state.apply_move(move)

                # Get the largest heuristic value
                value = max(
                    value,
                    self.abp(child, a, b, not max_turn, depth - 1)[0])

                # If the largest heuristic value is larger than alpha
                if value > a:
                    # Update alpha and the best move
                    a = value
                    best = move

                # If the best move is seemingly at least as good as what our
                # opponent can produce
                if a >= b:
                    return (value, best) # Return the value and the move

            # Return back the value and the best move we found
            return (value, best)

        # It's MIN's turn
        else:
            # A really high number
            value = np.inf

            # This is the same as MAX's turn but trying to find a small number
            # instead
            for move in moves:
                child = state.apply_move(move)

                value = min(
                    value,
                    self.abp(child, a, b, not max_turn, depth - 1)[0])

                if value < b:
                    b = value
                    best = move

                if a >= b:
                    return (value, move)

            return (value, move)

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        # Get the total number of pieces on the board
        total_count = 64 - sum([row.count("empty") for row in state.board])

        # If we are still in the early game or are in the endgame run alpha-beta
        if total_count < 20 or total_count > 40:
            return self.abp(state, -np.inf, np.inf, True, 5)[1]

        # Let MCTS handle the midgame to save some time
        return self.mcts(state, time.time() + (remaining_time // 15))


class TournamentPlayer(OthelloPlayer):
    """You should implement this class as your entry into the AI Othello tournament.
    You should implement other OthelloPlayers to try things out during your
    experimentation, but this is the only one that will be tested against your
    classmates' players."""

    def normalizer(self, max_value, min_value):
        """ Normalizes values calculated from heuristics for better weighing."""
        # As long as we have a nonzero sum
        if (max_value + min_value) != 0:
            # Normalize the values to some non-trivial amount
            return 100 * (max_value - min_value) / (max_value + min_value)

        # Zero is zero
        return 0

    def corner_check(self, state, plrX, plrN):
        """ Returns a normalized value for corner evalution of the board. """
        # Tracking the scores using player name as key
        scores = { plrX : 0, plrN : 0 }

        # Each list in the list represent 1 corner combination of 4 squares -
        # [0]: actual corner, [1]: diagonal squares, [2/3]: adjacent squares
        corners = [
            [ # The top left corner
                state.board[0][0],
                state.board[1][1],
                state.board[0][1],
                state.board[1][0]],
            [ # The bottom left corner
                state.board[7][0],
                state.board[6][1],
                state.board[7][1],
                state.board[6][0]],
            [ # The bottom right corner
                state.board[7][7],
                state.board[6][6],
                state.board[7][6],
                state.board[6][7]],
            [ # The top right corner
                state.board[0][7],
                state.board[1][6],
                state.board[0][6],
                state.board[1][7]]]

        # Check patterns for each corner
        for corner in corners:
            # If the corner is empty
            if corner[0] == "empty":
                # For indices [1, 3]
                for i in range(1,4):
                    # Punish placing around an uncaptured corner
                    if corner[i] != "empty":
                        scores[opposite_color(corner[i])] += 3

            # If the corner is captued:
            else:
                # Award capturing all 4 squares at the corner
                if len(set(corner)) == 1:
                    scores[corner[0]] += 15
                # Not all four corner squares are captured
                else:
                    corner_color = corner[0]
                    # Award capturing the corner
                    scores[corner_color] += 4
                    for i in range(1,4):
                        # Punish placing around a captured opponent corner
                        if corner[i] != corner_color and corner[i] != "empty":
                            scores[corner_color] += 2

                        # Award placing around a captured self corner
                        if corner[i] == corner_color and corner[i] != "empty":
                            scores[corner_color] += 3

        # Normalize the data and return it
        return normalizer(scores[plrX], scores[plrN])

    def mobility(self, max_moves, min_moves):
        """ Find the mobility score for the current board position """
        # The number of MAX and MIN moves
        moveX = len(max_moves)
        moveN = len(min_moves)

        # Return the mobility score of the position
        return self.normalizer(moveX, moveN)

    def stability(self, state, max_moves, min_moves, plrX, plrN):
        """ Find the stability score for the current board position """
        # The number of unstable pieces for MAX and MIN
        unstableX = 0
        unstableN = 0

        # For every move in MIN's possible moves
        for move in min_moves:
            (row, col) = move.pair

            # Directions in which MAX could lose control
            directions = []

            # For every direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # If there won't be any change
                    if dr==0 and dc==0:
                        continue

                    # If MAX is flanked by MIN in that direction
                    if state.flanking(row + dr, col + dc, dr, dc, plrX, plrN):
                        # Add the direction to the list
                        directions.append((dr, dc))

            # For each volatile direction...
            for dr, dc in directions:
                # Count all the pieces that are flippable
                while in_bounds(row + dr, col + dc) and \
                    state.board[row + dr][col + dc] == plrX:
                    unstableX += 1

        # For every move in MAX's possible moves
        for move in max_moves:
            (row, col) = move.pair

            # Directions in which MAX could lose control
            directions = []

            # For every direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # If there won't be any change
                    if dr==0 and dc==0:
                        continue

                    # If MIN is flanked by MAX in that direction
                    if state.flanking(row + dr, col + dc, dr, dc, plrN, plrX):
                        # Add the direction to the list
                        directions.append((dr, dc))

            # For each volatile direction...
            for dr, dc in directions:
                # Count all the pieces that are flippable
                while in_bounds(row + dr, col + dc) and \
                    state.board[row + dr][col + dc] == plrX:
                    unstableN += 1

        # The number of stable pieces for MAX and MIN
        stableX = state.count(plrX) - unstableX
        stableN = state.count(plrN) - unstableN

        # Return the stability score of the position
        return self.normalizer(stableX + unstableN, stableN + unstableX)

    def mobiliy_and_stability_check(self, state, plrX, plrN):
        """ Check the mobility and stability of the players in a given state """
        # Find all of MAX's moves
        max_moves = state.available_moves()

        # Create a deep copy of the state and force it to be MIN's turn
        min_state = copy.deepcopy(state)
        min_state.current = plrN

        # Find all of MIN's moves
        min_moves = min_state.available_moves()

        # Return the pair of mobility and stability scores
        return self.mobility(max_moves, min_moves), \
        self.stability(state, max_moves, min_moves, plrX, plrN)

    def heuristic(self, state, current_player):
        """ Given the state of Othello, return an evalution. """
        # For easier access of player colors
        plrX = current_player # for Ma"X"
        plrN = opposite_color(current_player) # for Mi"N"

        # Difference between players' pieces
        captures = self.normalizer(state.count(plrX), state.count(plrN))

        # Compare players' mobility and stability
        mobi, stabi = self.mobiliy_and_stability_check(state, plrX, plrN)

        # Compare the corner patterns
        corner = self.corner_check(state, plrX, plrN)

        # Multipliers/weights for different elements
        # Inspired by Paul S. Rosenbloom
        total_moves = 64 - sum([row.count("empty") for row in state.board])

        # Weights that we can tune for a better evaluation
        cap_w = 30
        m_w = (2 * total_moves) if total_moves < 30 else (total_moves)
        cor_w = 7 * total_moves
        s_w = 36

        # Apply the weights and return the sum of the factors discovered above
        return cap_w * captures + m_w * mobi + cor_w * corner + s_w * stabi

    def sim_game(self, state):
        """ Randomly simulate a game from the given state, returning the winner
        of the random game """
        # If the game is over, return the outcome of the game
        if state.game_over():
            return state.winner()

        # Get a random, legal move in the game
        move = random.choice(state.available_moves())

        # Recursively pick random moves until the game ends
        return self.sim_game(state.apply_move(move))

    def mcts(self, state, move_time):
        """ Perform MCTS on the current state to find a decent-looking move """
        # The parent for the new nodes
        parent = None

        # A placeholder for the tree we will be searching later
        tree = SearchTree(0, state, None)

        # Check for the edge case that the game is over
        if state.game_over():
            return None

        # While we still have time
        while move_time - time.time() > 0:
            # Get the most interesting leaf
            node = tree.traverse()

            # Simulate the game using the most promising move found so far
            outcome = self.sim_game(node.state)

            # Get the win value
            if outcome == 'draw':
                win = .5
            elif outcome == self.color:
                win = 1
            else:
                win = 0

            # Update the values in the tree
            node.update(win)

        # Store the best node and its rating
        best_child, best_confidence = None, -np.inf

        # For every child
        for child in tree.children:
            # Discard children with no plays
            if child.plays < 1:
                continue

            # Track the average heuristic value of the node's children
            avg_h = 0

            # For every child of the child (grandchild?)
            for next_child in child.children:
                # Divide the heuristic value of the grandchild by the
                # number of grandchildren and add it to the average total
                avg_h += self.heuristic(next_child.state, self.color) / \
                len(child.children)

            # Get some measure of how well the child will perform
            child_confidence = (child.plays / math.sqrt(2)) + avg_h

            # If this one is better than the others
            if child_confidence > best_confidence:
                best_child, best_confidence = child, child_confidence

        # Return the best child
        return state.available_moves()[tree.children.index(best_child)]

    def abp(self, state, a, b, max_turn, depth=1):
        """ The standard Alpha-beta pruning algorithm """
        # Get all the available moves
        moves = state.available_moves()

        # If there are no available moves (i.e. terminal node) or we have hit
        # the maximum depth
        if moves == [] or depth < 1:
            return (self.heuristic(state, self.color), None)

        # Track the best move so far
        best = None

        if max_turn:
            # A very low number
            value = -np.inf

            # For every available move
            for move in moves:
                child = state.apply_move(move)

                # Get the largest heuristic value
                value = max(
                    value,
                    self.abp(child, a, b, not max_turn, depth - 1)[0])

                # If the largest heuristic value is larger than alpha
                if value > a:
                    # Update alpha and the best move
                    a = value
                    best = move

                # If the best move is seemingly at least as good as what our
                # opponent can produce
                if a >= b:
                    return (value, best) # Return the value and the move

            # Return back the value and the best move we found
            return (value, best)

        # It's MIN's turn
        else:
            # A really high number
            value = np.inf

            # This is the same as MAX's turn but trying to find a small number
            # instead
            for move in moves:
                child = state.apply_move(move)

                value = min(
                    value,
                    self.abp(child, a, b, not max_turn, depth - 1)[0])

                if value < b:
                    b = value
                    best = move

                if a >= b:
                    return (value, move)

            return (value, move)

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. """
        # Get the total number of pieces on the board
        total_count = 64 - sum([row.count("empty") for row in state.board])

        # If we are still in the early game or are in the endgame run alpha-beta
        if total_count < 20 or total_count > 40:
            return self.abp(state, -np.inf, np.inf, True, 5)[1]

        # Let MCTS handle the midgame to save some time
        return self.mcts(state, time.time() + (remaining_time // 15))


def in_bounds(row, col):
    """ True if the pair is within the bounds of the board """
    if row > 7 or row < 0:
        return False
    if col > 7 or col < 0:
        return False


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
    corners = [
        state.board[0][0], # Top left
        state.board[7][7], # Bottom Right
        state.board[0][7], # Top Right
        state.board[7][0]] # Bottom Left

    # Normalize the data for the number of times MAX has the corner
    actual_corners = normalizer(corners.count(plrX), corners.count(plrN))

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
    # Get the number of moves MAX can make
    moveX = len(state.available_moves())

    # Make a deep copy of the state and force MIN to be the current player
    min_state = copy.deepcopy(state)
    min_state.current = plrN

    # Get the number of moves MIN can make
    moveN = len(min_state.available_moves())

    # Normalize the player moves
    moves = normalizer(moveX, moveN)

    return count + corners + moves


def static_heuristic(state, color):
    """ A static heuristic function that lables each part of the board
        with a constant value. Proven comparable to simple_heuristic, but
        not better than others since it cannot dynamically evaluate. """
    # Get the player colors in a state that's easier to access
    plrX = color # For MA(X)
    plrN = opposite_color(color) # For MI(N)

    # Apply weights to the board
    score_board = [
        [40, -3, 2, 2, 2, 2, -3, 40],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [2, -1, 1, 0, 0, 1, -1, 2],
        [2, -1, 0, 1, 1, 0, -1, 2],
        [2, -1, 0, 1, 1, 0, -1, 2],
        [2, -1, 1, 0, 0, 1, -1, 2],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [40, -3, 2, 2, 2, 2, -3, 40]]

    # The weighted score for the current board position
    plrX_score = 0
    plrN_score = 0

    # For every row and col in the board
    for row in range(8):
        for col in range(8):
            # Get the player at that position (NOTE: could be 'empty')
            place = state.board[row][col]

            # If the player is MAX
            if place == plrX:
                plrX_score += score_board[row][col]

            # If the player is MIN
            if place == plrN:
                plrN_score += score_board[row][col]

    # Subtract MIN's weighted score from MAX's weighted score
    return plrX_score - plrN_score


def normalizer(max_value, min_value):
    """ Normalizes values calculated from heuristics for better weighing."""
    # As long as we have a nonzero sum
    if (max_value + min_value) != 0:
        # Normalize the values to some non-trivial amount
        return 100 * (max_value - min_value) / (max_value + min_value)

    # Zero is zero
    return 0


def corner_check(state, plrX, plrN):
    """ Returns a normalized value for corner evalution of the board. """
    # Tracking the scores using player name as key
    scores = { plrX : 0, plrN : 0 }

    # Each list in the list represent 1 corner combination of 4 squares -
    # [0]: actual corner, [1]: diagonal squares, [2/3]: adjacent squares
    corners = [
        [ # The top left corner
            state.board[0][0],
            state.board[1][1],
            state.board[0][1],
            state.board[1][0]],
        [ # The bottom left corner
            state.board[7][0],
            state.board[6][1],
            state.board[7][1],
            state.board[6][0]],
        [ # The bottom right corner
            state.board[7][7],
            state.board[6][6],
            state.board[7][6],
            state.board[6][7]],
        [ # The top right corner
            state.board[0][7],
            state.board[1][6],
            state.board[0][6],
            state.board[1][7]]]

    # Check patterns for each corner
    for corner in corners:
        # If the corner is empty
        if corner[0] == "empty":
            # For indices [1, 3]
            for i in range(1,4):
                # Punish placing around an uncaptured corner
                if corner[i] != "empty":
                    scores[opposite_color(corner[i])] += 3

        # If the corner is captued:
        else:
            # Award capturing all 4 squares at the corner
            if len(set(corner)) == 1:
                scores[corner[0]] += 15
            # Not all four corner squares are captured
            else:
                corner_color = corner[0]
                # Award capturing the corner
                scores[corner_color] += 4
                for i in range(1,4):
                    # Punish placing around a captured opponent corner
                    if corner[i] != corner_color and corner[i] != "empty":
                        scores[corner_color] += 2

                    # Award placing around a captured self corner
                    if corner[i] == corner_color and corner[i] != "empty":
                        scores[corner_color] += 3

    # Normalize the data and return it
    return normalizer(scores[plrX], scores[plrN])


def end_game_heuristic(state, color):
    """ An attempt to improve from the advanced_heuristic. """
    #### For easier access of player colors
    plrX = color # for Ma"X"
    plrN = opposite_color(color) # for Mi"N"

    #### Difference between players' piece
    captures = normalizer(state.count(plrX), state.count(plrN))

    #### Difference between players' mobilities
    # Get the number of moves MAX can make
    moveX = len(state.available_moves())

    # Make a deep copy of the state and force MIN to be the current player
    min_state = copy.deepcopy(state)
    min_state.current = plrN

    # Get the number of moves MIN can make
    moveN = len(min_state.available_moves())
    mobility = normalizer(moveX, moveN)

    #### Compare the corner patterns
    corner  = corner_check(state, plrX, plrN)

    #### Compare stability between the players
    total_moves = 64 - sum([row.count("empty") for row in state.board])
    mobi_multi = 2 * total_moves if total_moves < 30 else total_moves

    # Weight the factors from above and return their sum
    return 30 * captures + mobi_multi * mobility + 6 * total_moves * corner


def helmuth_heuristic(state, color):
    """ The heuristic used by Professor T. Helmuth, used for debug only. """
    return state.count(color) - state.count(opposite_color(color))

################################################################################

def main():
    """Plays the game."""
    # black_player = hw2.ShortTermMaximizer("black")
    # white_player = hw2.ShortTermMaximizer("white")
    # black_player = MinimaxPlayer("black", helmuth_heuristic)
    # white_player = MinimaxPlayer("white", helmuth_heuristic)
    # black_player = AlphaBetaPruner("black", advanced_heuristic)
    # white_player = AlphaBetaPruner("white", advanced_heuristic)
    # black_player = MCTSPlayer("black", advanced_heuristic)
    # white_player = MCTSPlayer("white", advanced_heuristic)

    # black_player = T1Player("black")
    # white_player = T1Player("white")
    black_player = T2Player("black")
    # white_player = T2Player("white")

    # black_player = HumanPlayer("black")
    white_player = HumanPlayer("white")

    game = OthelloGame(black_player, white_player, verbose=True)

    winner = game.play_game()

    print("black (X): {}, time remains: {}".format(
        game.board.count('black'), game.black_time))
    print("white (O): {}, time remains: {}".format(
        game.board.count('white'), game.white_time))
    print(game.board)

    if not game.verbose:
        print("Winner is", winner)


if __name__ == "__main__":
    main()
