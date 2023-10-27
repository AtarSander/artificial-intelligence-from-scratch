from tabulate import tabulate
from gradientdescent import GradientDescent
import numpy as np


def experiment(function, gradient, x0, step_size, iterations):
    sample = GradientDescent(step_size, iterations)
    return sample.solve(gradient, x0, function)


def experiment_serie(domain, function, gradient,
                     step_sizes, iteration_values):

    start_points = np.random.uniform(domain[0], domain[1], 10)
    gradient_results_x = []
    gradient_results_y = []

    for step_size in step_sizes:
        for iteration_value in iteration_values:
            for start_point in start_points:
                min_x, min_y = experiment(function, gradient, start_point,
                                          step_size, iteration_value)
                gradient_results_x.append(min_x)
                gradient_results_y.append(min_y)

    num_of_ex = [i for i in range(1, (len(step_sizes) * len(iteration_values) * len(start_points)) + 1)]
    iter_values_formatted = list(map(lambda x: [x] * len(start_points), iteration_values))
    iter_values_formatted = [item for sublist in iter_values_formatted for item in sublist]

    step_sizes_formatted = list(map(lambda x: [x] * len(start_points)* len(iteration_values), step_sizes))
    step_sizes_formatted = [item for sublist in step_sizes_formatted for item in sublist]

    experiments_table = {"Experiment number": num_of_ex, "Step size": step_sizes_formatted,
                         "Number of iterations": iter_values_formatted * len(step_sizes),
                         "Starting points": list(start_points) * len(iteration_values) * len(step_sizes),
                         "End x": gradient_results_x, "End y": gradient_results_y}

    return tabulate(experiments_table, headers="keys", tablefmt="fancy_grid")
