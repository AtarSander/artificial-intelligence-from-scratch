from abc import ABC, abstractmethod
from heuristic import heuristic
from two_player_games.game import Game, Player, Move
import numpy as np
import random
import copy


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
    def __init__(self, depth: int):
        self.depth = depth

    def get_parameters(self) -> int:
        """Returns a dictionary of hyperparameters"""
        return self.depth

    def solve(self, game: Game, depth: int, alpha: int, beta: int, maximizing_player: Player) -> tuple[int, Move]:
        if game.is_finished():
            if game.get_winner() is None:
                return None, 0
            if game.get_winner().char == maximizing_player.char:
                return None, np.inf
            else:
                return None, -np.inf
        if depth == 0:
            return None, heuristic(game.state.fields, maximizing_player)

        if maximizing_player.char == game.get_current_player().char:
            max_value = -np.inf
            same_quality = []
            for move in game.get_moves():
                copy_game = copy.deepcopy(game)
                copy_game.make_move(move)
                value = self.solve(copy_game, depth-1, alpha, beta, maximizing_player)[1]
                if value > max_value:
                    max_value = value
                    same_quality = [(move, value)]
                elif value == max_value:
                    same_quality.append((move, value))
                alpha = max(alpha, max_value)
                if max_value >= beta:
                    break
            return random.choice(same_quality)

        else:
            min_value = np.inf
            same_quality = []
            for move in game.get_moves():
                copy_game = copy.deepcopy(game)
                copy_game.make_move(move)
                value = self.solve(copy_game, depth-1, alpha, beta, maximizing_player)[1]
                if value < min_value:
                    min_value = value
                    same_quality = [(move, value)]
                elif value == min_value:
                    same_quality.append((move, value))
                beta = min(beta, min_value)
                if min_value <= alpha:
                    break
            return random.choice(same_quality)
