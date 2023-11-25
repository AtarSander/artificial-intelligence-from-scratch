from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, node, *args, **kwargs):
        """
        A method that solves the given problem for given initial solutions.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...


class Alphabeta(Solver):
    def __init__(self, depth):
        self.depth = depth

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return self.depth

    def solve(self, game, depth, alpha, beta, maximizingPlayer):
        if depth == 0:
            return None, 0
        if game.is_finished():
            if game.get_winner() == maximizingPlayer:
                return None, 1
            elif game.get_winner() is None:
                return None, 0
            else:
                return None, -1

        if maximizingPlayer == game.get_current_player():
            max_value = -np.inf
            max_move = None
            for move in game.get_moves():
                game.make_move(move)
                value = self.solve(game, depth-1, alpha, beta, game.get_current_player())[1]
                if value > max_value:
                    max_value = value
                    max_move = move
                alpha = max(alpha, max_value)
                if max_value >= beta:
                    break
            return max_move, max_value

        else:
            min_value = np.inf
            min_move = None
            for move in game.get_moves():
                game.make_move(move)
                value = self.solve(game, depth-1, alpha, beta, game.get_current_player())[1]
                if value < min_value:
                    min_value = value
                    min_move = move
                beta = min(beta, min_value)
                if min_value <= alpha:
                    break
            return min_move, min_value
