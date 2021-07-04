"""
Author:         Nate Koike & Zhaosen Guo
Date:           2021/3/1
Description:    Implement agents to play Othello with different strategies.
"""

import random
from othello import *

class MoveNotAvailableError(Exception):
    """Raised when a move isn't available. Do not change."""
    pass


class OthelloPlayer():
    """Parent class for Othello players. Do not change."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass


class HumanPlayer(OthelloPlayer):
    """Allows a human to play the game."""

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


class RandomPlayer(OthelloPlayer):
    """AI Player that moves randomly"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""

        # Get all available moves and return a random choice
        return random.choice(state.available_moves())


class ShortTermMaximizer(OthelloPlayer):
    """AI Player that maximizes the number of their color pieces on the board
    after that move"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""

        # Covert all the possible moves to OthelloMove objects
        moves = state.available_moves()

        # Initate the counter for the number of player's pieces
        max_move = None
        max_count = 0

        # For every available move
        for move in moves:
            # Get the number of pieces that are the same color as the player
            new_count = copy.deepcopy(state).apply_move(move).count(self.color)

            # If the new count is greater
            if new_count > max_count:
                # Save the move and count
                max_move, max_count = move, new_count

        return max_move


class ShortTermMinimizer(OthelloPlayer):
    """AI Player that minimizes the number of their color pieces on the board
    after that move"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""

        # Covert all the possible moves to OthelloMove objects
        moves = state.available_moves()

        # Initate the counter for the number of player's pieces
        min_move = moves.pop()
        min_count = copy.deepcopy(state).apply_move(min_move).count(self.color)

        # For every available move
        for move in moves:
            # Get the number of pieces that are the same color as the player
            new_count = copy.deepcopy(state).apply_move(move).count(self.color)

            # If the new count is greater
            if new_count < min_count:
                # Save the move and count
                min_move, min_count = move, new_count

        return min_move


class MaximizeOpponentMoves(OthelloPlayer):
    """AI Player that maximizes the number of moves its opponent will have"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""

        # Covert all the possible moves to OthelloMove objects
        moves = state.available_moves()

        # Initate the counter for the number of player's pieces
        max_move = moves.pop()
        max_moves = len(copy.deepcopy(state).apply_move(max_move).available_moves())

        # For every available move
        for move in moves:
            # Get the number of moves available to the opponent after applying
            # the current move
            new_moves = len(copy.deepcopy(state).apply_move(move).available_moves())

            # If the opponent has more available moves
            if new_moves > max_moves:
                # Save the move and the next number of available moves
                max_move, max_moves = move, new_moves

        return max_move


class MinimizeOpponentMoves(OthelloPlayer):
    """AI Player that minimizes the number of moves its opponent will have"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""

        # Covert all the possible moves to OthelloMove objects
        moves = state.available_moves()

        # Initate the counter for the number of player's pieces
        min_move = moves.pop()
        min_moves = len(copy.deepcopy(state).apply_move(min_move).available_moves())

        # For every available move
        for move in moves:
            # Get the number of moves available to the opponent after applying
            # the current move
            new_moves = len(copy.deepcopy(state).apply_move(move).available_moves())

            # If the opponent has fewer available moves
            if new_moves < min_moves:
                # Save the move and the next number of available moves
                min_move, min_moves = move, new_moves

        return min_move


################################################################################

def main():
    """Plays the game. You'll likely want to make a new main function for
    playing many games using your players to gather stats."""

    # black_player = HumanPlayer("black")
    white_player = HumanPlayer("white")

    black_player = MinimizeOpponentMoves("black")
    # white_player = MinimizeOpponentMoves("white")

    game = OthelloGame(black_player, white_player, verbose=True)

    ###### Use this method if you want to play a timed game. Doesn't work with HumanPlayer
    # winner = game.play_game()

    ###### Use this method if you want to use a HumanPlayer
    #winner = game.play_game()

    ###### Use this method if you want to collect extra timing data
    winner = game.play_game_data()


    if not game.verbose:
        print("Winner is", winner)


if __name__ == "__main__":
    main()
