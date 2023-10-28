from tabulate import tabulate
from gradientdescent import GradientDescent
import numpy as np


def experiment_serie(start_points, function, gradient,
                     step_sizes, iteration_values, correct_value):

    start_points = np.array(start_points, dtype=np.float128)
    gradient_results_x = []
    gradient_results_y = []
    error_x = []
    get_first = GradientDescent(step_sizes[0], iteration_values[0])
    max_x, _ = get_first.solve(gradient, start_points[0], function)
    worst_grad_steps_x = get_first.get_parameters()["X_values"]
    worst_grad_steps_y = get_first.get_parameters()["Y_values"]
    min_x = start_points[0]

    for step_size in step_sizes:
        for iteration_value in iteration_values:
            for start_point in start_points:
                sample = GradientDescent(step_size, iteration_value)
                x, y = sample.solve(gradient, start_point, function)
                gradient_results_x.append(x)
                gradient_results_y.append(y)
                error_x.append(np.linalg.norm(np.subtract(x, correct_value)))
                if np.linalg.norm(np.subtract(x, correct_value)) < np.linalg.norm(min_x):
                    min_x = x
                    best_grad_steps_x = sample.get_parameters()["X_values"]
                    best_grad_steps_y = sample.get_parameters()["Y_values"]
                if np.linalg.norm(np.subtract(x, correct_value)) > np.linalg.norm(max_x):
                    max_x = x
                    worst_grad_steps_x = sample.get_parameters()["X_values"]
                    worst_grad_steps_y = sample.get_parameters()["Y_values"]


    results = (step_sizes, iteration_values, start_points, gradient_results_x, gradient_results_y, error_x)
    best = (best_grad_steps_x, best_grad_steps_y)
    worst = (worst_grad_steps_x, worst_grad_steps_y)

    return results, best, worst


def create_table(step_sizes, iteration_values, start_points,
                 gradient_results_x, gradient_results_y, error_x):

    num_of_ex = [i for i in range(1, (len(step_sizes) * len(iteration_values) * len(start_points)) + 1)]

    iter_values_formatted = list(map(lambda x: [x] * len(start_points), iteration_values))
    iter_values_formatted = [item for sublist in iter_values_formatted for item in sublist]

    step_sizes_formatted = list(map(lambda x: [x] * len(start_points) * len(iteration_values), step_sizes))
    step_sizes_formatted = [item for sublist in step_sizes_formatted for item in sublist]

    experiments_table = {"Experiment number": num_of_ex, "Step size": step_sizes_formatted,
                         "Number of iterations": iter_values_formatted * len(step_sizes),
                         "Starting points": list(start_points) * len(iteration_values) * len(step_sizes),
                         "End x": gradient_results_x, "End y": gradient_results_y, "Error value": error_x}

    return tabulate(experiments_table, headers="keys", tablefmt="fancy_grid")