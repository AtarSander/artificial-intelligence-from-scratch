from genetic import Genetic
from itertools import product
from physics import target_function
import numpy as np


def satisfy_budget(ind_num, iter_nums, budget):
    ind_and_iter = []
    for i in ind_num:
        for j in iter_nums:
            if i*j == budget:
                ind_and_iter.append((i, j))
    return ind_and_iter


def experiment(ind_num, iter_nums, cross_rate, mut_rate, budget):
    ind_and_iter = satisfy_budget(ind_num, iter_nums, budget)
    combinations = list(product(ind_and_iter, cross_rate, mut_rate))
    g_mean = []
    g_std = []
    g_stats = []
    g_stats_std = []

    solution = Genetic()

    for combo in combinations:
        g = []
        g_runs = []
        for _ in range(25):
            x_best, g_best = solution.solve(target_function, combo[0][0],
                                            combo[0][1], combo[1], combo[2])
            g_runs.append(solution.get_best_grades())
            g.append(g_best)
        g_mean.append(np.mean(g))
        g_std.append(round(np.std(g), 2))
        g_stats.append(np.mean(g_runs, axis=0))
        g_stats_std.append(np.std(g_runs, axis=0))

    return g_mean, g_std, g_stats, g_stats_std


def create_table(ind_and_iter, cross_rate, mut_rate, g_mean, g_std):

    num_of_ex = [i for i in range(1, (len(ind_and_iter) *
                                      len(cross_rate) * len(mut_rate)) + 1)]
    result = [str(g_mean[i]) + "+-" + str(g_std[i])
              for i in range(len(g_mean))]

    individuals = [i[0] for i in ind_and_iter]
    individuals_formatted = list(map(lambda x: [x] * len(cross_rate) *
                                     len(mut_rate), individuals))
    individuals_formatted = [item for sublist in individuals_formatted
                             for item in sublist]

    iterations = [i[1] for i in ind_and_iter]
    iterations_formatted = list(map(lambda x: [x] * len(cross_rate) *
                                    len(mut_rate), iterations))
    iterations_formatted = [item for sublist in iterations_formatted
                            for item in sublist]

    cross_rate_formatted = list(map(lambda x: [x] * len(mut_rate), cross_rate))
    cross_rate_formatted = [item for sublist in cross_rate_formatted
                            for item in sublist]

    experiments_table = {"Experiment number": num_of_ex,
                         "Individuals number": individuals_formatted,
                         "Iterations number": iterations_formatted,
                         "Cross probability": cross_rate_formatted * len(iterations),
                         "Mutation probability": mut_rate * len(cross_rate) * len(ind_and_iter),
                         "Mean best result (25 runs)": result}
    return experiments_table
