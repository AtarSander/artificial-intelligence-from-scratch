from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, x0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """


class GradientDescent(Solver):
    def __init__(self, step_size, iterations, epsilon):
        self.step_size = step_size
        self.iterations = iterations
        self.epsilon = epsilon
        self.x_values = []
        self.y_values = []

    def get_parameters(self):
        dict = {}
        dict["Learning_rate"] = self.step_size
        dict["Iterations"] = self.iterations
        dict["X_values"] = self.x_values
        dict["Y_values"] = self.y_values
        dict["Epsilon"] = self.epsilon
        return dict

    def set_learning_rate(self, new_step_size):
        self.step_size = new_step_size

    def set_iterations(self, new_iter_num):
        self.iterations = new_iter_num

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def solve(self, problem, x0, funct, epsilon):
        n_iter = self.iterations

        while n_iter > 0 and x0 > epsilon:
            self.x_values.append(x0)
            self.y_values.append(funct(x0))
            n_iter -= 1
            d = problem(x0)
            x0 = np.subtract(x0, d * self.step_size)
        return x0
