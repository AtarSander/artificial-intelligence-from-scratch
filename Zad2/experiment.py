from genetic import Genetic
from itertools import product
from physics import target_function


def experiment(ind_num, iter_nums, cross_rate, mut_rate, budget):
    ind_and_iter = []
    for i in ind_num:
        for j in iter_nums:
            if i*j == budget:
                ind_and_iter.append((i, j))

    combinations = list(product(ind_and_iter, cross_rate, mut_rate))
    solution = Genetic()
    g = []
    for combo in combinations:
        o = 0
        for _ in range(25):
            x_best, o_best = solution.solve(target_function, combo[0][0],
                                            combo[0][1], combo[1], combo[2])
            o += o_best
        o /= 25
        g.append(o)
    return g
