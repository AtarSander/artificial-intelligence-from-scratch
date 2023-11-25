from alphabeta import Alphabeta
import sys
import random
import numpy as np
import copy
sys.path.append("two-player-games")
from two_player_games.games.connect_four import ConnectFour


def experiment(type, games_num, ab_turn=None, depths=None):
    results = {}
    if depths is not None:
        results["Number of games"] = [games_num for _ in range(len(depths))]
    else:
        results["Number of games"] = [games_num]
    if type == 1:
        player_1_wins = 0
        player_2_wins = 0
        draws = 0
        for _ in range(games_num):
            game = ConnectFour()

            while not game.is_finished():
                moves = game.get_moves()
                move = random.choice(moves)
                game.make_move(move)
            winner = game.get_winner()

            if winner is None:
                draws += 1
            elif winner.char == '1':
                player_1_wins += 1
            else:
                player_2_wins += 1
        results["Player 1 wins"] = [player_1_wins]
        results["Player 2 wins"] = [player_2_wins]
        results["Draws"] = [draws]

    elif type == 2 and depths and ab_turn:
        player_1_wins_depth = []
        player_2_wins_depth = []
        draws_depth = []
        for depth in depths:
            player_1_wins = 0
            player_2_wins = 0
            draws = 0
            for _ in range(games_num):
                game = ConnectFour()
                ab_player = Alphabeta(depth)

                winner = ab_game(game, ab_player, ab_turn, depth)

                if winner is None:
                    draws += 1
                elif winner.char == '1':
                    player_1_wins += 1
                else:
                    player_2_wins += 1

            player_1_wins_depth.append(player_1_wins)
            player_2_wins_depth.append(player_2_wins)
            draws_depth.append(draws)
        results["Player 1 wins"] = player_1_wins_depth
        results["Player 2 wins"] = player_2_wins_depth
        results["Draws"] = draws_depth
        results["Depths"] = depths

    else:
        raise AttributeError

    return results


def ab_game(game, ab_player, who, depth):
    if who == 1:
        max_player = game.get_current_player()
        while not game.is_finished():
            temp_game = copy.deepcopy(game)
            move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
            game.make_move(move_ab)
            if not game.is_finished():
                moves = game.get_moves()
                move_rand = random.choice(moves)
                game.make_move(move_rand)
        winner = game.get_winner()

    elif who == 2:
        max_player = game.get_current_player()
        while not game.is_finished():
            moves = game.get_moves()
            move_rand = random.choice(moves)
            game.make_move(move_rand)
            if not game.is_finished():
                temp_game = copy.deepcopy(game)
                move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
                game.make_move(move_ab)
        winner = game.get_winner()

    elif who == 3:
        max_player = game.get_current_player()
        while not game.is_finished():
            temp_game = copy.deepcopy(game)
            move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
            game.make_move(move_ab)
            if not game.is_finished():
                temp_game = copy.deepcopy(game)
                move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
                game.make_move(move_ab)
        winner = game.get_winner()
    else:
        raise AttributeError
    return winner
