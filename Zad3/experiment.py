from alphabeta import Alphabeta
import sys
import random
import numpy as np
import copy
from enum import IntEnum
sys.path.append("Zad3/two-player-games")
from two_player_games.games.connect_four import ConnectFour


class Option(IntEnum):
    RAND_VS_RAND = 1
    ALGO_VS_RAND = 2
    RAND_VS_ALGO = 3
    ALGO_VS_ALGO = 4
    ALGO_VS_HUMAN = 5
    HUMAN_VS_ALGO = 6


def experiment(option, games_num, depths=None):
    results = {}
    if option == 1:
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

        results["Number of games"] = [games_num]
        results["Player 1 wins"] = [player_1_wins]
        results["Player 2 wins"] = [player_2_wins]
        results["Draws"] = [draws]

    elif option == 2 or option == 3:
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

                winner = ab_game(game, ab_player, option, depth)

                if winner is None:
                    draws += 1
                elif winner.char == '1':
                    player_1_wins += 1
                else:
                    player_2_wins += 1

            player_1_wins_depth.append(player_1_wins)
            player_2_wins_depth.append(player_2_wins)
            draws_depth.append(draws)
        results["Number of games"] = [games_num for _ in range(len(depths))]
        results["Player 1 wins"] = player_1_wins_depth
        results["Player 2 wins"] = player_2_wins_depth
        results["Draws"] = draws_depth
        results["Depths"] = depths

    elif option == 4:
        player_1_wins_depth = []
        player_2_wins_depth = []
        draws_depth = []
        for depth1 in depths:
            for depth2 in depths:
                player_1_wins = 0
                player_2_wins = 0
                draws = 0
                for _ in range(games_num):
                    game = ConnectFour()
                    ab_player = Alphabeta(depth1)

                    winner = ab_game(game, ab_player, option, depth1, depth2)

                    if winner is None:
                        draws += 1
                    elif winner.char == '1':
                        player_1_wins += 1
                    else:
                        player_2_wins += 1

                player_1_wins_depth.append(player_1_wins)
                player_2_wins_depth.append(player_2_wins)
                draws_depth.append(draws)
        results["Number of games"] = [games_num for _ in range(len(depths) * len(depths))]
        results["Player 1 wins"] = player_1_wins_depth
        results["Player 2 wins"] = player_2_wins_depth
        results["Draws"] = draws_depth
        results["Depths player 1"] = [depth for depth in depths for _ in range(len(depths))]
        results["Depths player 2"] = depths * len(depths)

    else:
        raise AttributeError("Wrong option chosen")

    return results


def ab_game(game, ab_player, who, depth, depth2=None):
    # Algo vs rand
    if who == 2:
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

    elif who == 3:
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

    # Algo vs algo
    elif who == 4:
        max_player = game.get_current_player()
        while not game.is_finished():
            temp_game = copy.deepcopy(game)
            move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
            game.make_move(move_ab)
            if not game.is_finished():
                temp_game = copy.deepcopy(game)
                move_ab = ab_player.solve(temp_game, depth2, -np.inf, np.inf, max_player)[0]
                game.make_move(move_ab)
        winner = game.get_winner()

    # Human vs player
    elif who == 5:
        max_player = game.get_current_player()
        while not game.is_finished():
            temp_game = copy.deepcopy(game)
            move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
            game.make_move(move_ab)
            if not game.is_finished():
                chosen = -1
                print(game)
                move = int(input("Choose column: "))
                moves = game.get_moves()
                for i in moves:
                    if i.column == move:
                        chosen = i
                if chosen != -1:
                    game.make_move(chosen)
                else:
                    raise AttributeError("Wrong move chosen")
        print(game)
        winner = game.get_winner()

    elif who == 6:
        max_player = game.get_current_player()
        while not game.is_finished():
            chosen = -1
            print(game)
            move = int(input("Choose column: "))
            moves = game.get_moves()
            for i in moves:
                if i.column == move:
                    chosen = i
            if chosen != -1:
                game.make_move(chosen)
            else:
                raise AttributeError("Wrong move chosen")
            if not game.is_finished():
                temp_game = copy.deepcopy(game)
                move_ab = ab_player.solve(temp_game, depth, -np.inf, np.inf, max_player)[0]
                game.make_move(move_ab)
        print(game)
        winner = game.get_winner()
    else:
        raise AttributeError("Wrong option chosen")
    return winner


def play_game(option, depth):
    game = ConnectFour()
    ab_player = Alphabeta(depth)
    if option == 5 or option == 6:
        winner = ab_game(game, ab_player, option, depth)
        if winner is None:
            result = "It's a draw!"
        elif winner.char == '1' and option == 5:
            result = "You lose!"
        elif winner.char == '2' and option == 5:
            result = "You win!"
        elif winner.char == '1' and option == 6:
            result = "You win!"
        else:
            result = "You lose!"
    else:
        raise AttributeError("Wrong option chosen")
    return result
