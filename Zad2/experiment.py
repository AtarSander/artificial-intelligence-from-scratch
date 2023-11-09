from genetic import Genetic
from itertools import product
from physics import target_function
import numpy as np


def experiment(ind_num, iter_nums, cross_rate, mut_rate):
    combinations = list(product(ind_num, iter_nums, cross_rate, mut_rate))
    solution = Genetic()
    for combo in combinations:
        g = []
        o = 0
        for _ in range(25):
            x_best, o_best = solution.solve(target_function, combo[0], combo[1], combo[2], combo[3])
            o += o_best
        o /= 25
        g.append(o)
    return g